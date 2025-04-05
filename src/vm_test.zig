// Test file for RISC-V virtual memory implementation

const std = @import("std");

const memory = @import("memory.zig");
const AddressSpace = memory.AddressSpace;
const MemoryMode = memory.MemoryMode;
const PhysicalPage = memory.PhysicalPage;
const PageTable = memory.PageTable;
const PTE_R = memory.PTE_R;
const PTE_W = memory.PTE_W;
const PTE_X = memory.PTE_X;
const PTE_U = memory.PTE_U;
const PTE_COW = memory.PTE_COW;
const riscv = @import("riscv.zig");
const RiscVCpu = riscv.RiscVCpu;
const vm_integration = @import("vm_integration.zig");
const MMUResources = vm_integration.MMUResources;

// zig test vm_test.zig -lc -I../deps/softfloat ../deps/softfloat/*.c

// Global variables for the ECALL hook test
var g_ecall_hook_called = false;
var g_last_syscall_num: u64 = 0;
var g_last_args: [6]u64 = [_]u64{0} ** 6;

// Global ECALL hook function
fn globalEcallHook(address_space: *AddressSpace, syscall_num: u64, args: [6]u64, user_data: ?*anyopaque) u64 {
    _ = address_space;
    _ = user_data;

    // Record that we were called and capture the arguments
    g_ecall_hook_called = true;
    // Record that we were called and capture the arguments
    g_ecall_hook_called = true;
    g_last_syscall_num = syscall_num;
    g_last_args = args;

    // For testing, just return the syscall number * 2
    return syscall_num * 2;
}

fn testCpuAddSharedMemory() !void {
    // process.resources.readMemory is global memory
    // address_space.resources.readMemory is per-address-space memory

    const allocator = std.testing.allocator;

    var process = try RiscVCpu.init(0, allocator);
    defer process.deinit(allocator);

    // Create a shared memory region
    const shared_vaddr: u64 = 0x1000;

    const space = try AddressSpace.init(allocator, .Sv39);
    defer space.deinit(process.resources.manager);

    try process.resources.mapMemory(shared_vaddr, 4096, PTE_R | PTE_W | PTE_X | PTE_U, space);

    // Write the instructions to memory
    try process.resources.writeMemory(u32, shared_vaddr, 0x00200513, space); // addi a0, zero, 2
    try process.resources.writeMemory(u32, shared_vaddr + 4, 0x00200593, space); // addi a1, zero, 2
    try process.resources.writeMemory(u32, shared_vaddr + 8, 0x00b50633, space); // add a2, a0, a1
    try process.resources.writeMemory(u32, shared_vaddr + 12, 0x00100073, space); // ebreak

    // Set up the stack pointer
    space.registers[2] = shared_vaddr + 4096 - 16; // sp = x2

    // Set the program counter to our code
    space.pc = shared_vaddr;

    var steps: usize = 0;
    const max_steps = 8;
    while (steps < max_steps) : (steps += 1) {
        // Step until ebreak
        std.debug.print("PC: 0x{x:016}, Instruction: 0x{x:08}\n", .{ space.pc, try process.resources.readMemory(u32, space.pc, space) });
        try process.executeInstruction(space, null);
        const inst = try process.resources.readMemory(u32, space.pc, space);
        if (inst == 0x00100073) {
            break;
        }
    }

    std.debug.print("Step 2: {}\n", .{space.registers[12]});

    const thread_space = try space.createThread();
    defer thread_space.deinit(process.resources.manager);

    // Set up the thread's program counter to our code
    thread_space.pc = shared_vaddr;

    var steps2: usize = 0;
    const max_steps2 = 8;
    while (steps2 < max_steps2) : (steps2 += 1) {
        // Step until ebreak
        std.debug.print("PC: 0x{x:016}, Instruction: 0x{x:08}\n", .{ thread_space.pc, try process.resources.readMemory(u32, thread_space.pc, thread_space) });
        try process.executeInstruction(thread_space, null);
        const inst = try process.resources.readMemory(u32, thread_space.pc, thread_space);
        if (inst == 0x00100073) {
            break;
        }
    }
}

// Test ECALL hook functionality
fn testEcallHook() !void {
    const allocator = std.testing.allocator;

    var cpu = try RiscVCpu.init(0, allocator);

    const address_space = try AddressSpace.init(allocator, .Sv39);

    defer cpu.deinit(allocator);
    defer address_space.deinit(cpu.resources.manager);

    // Reset global variables
    g_ecall_hook_called = false;
    g_last_syscall_num = 0;
    g_last_args = [_]u64{0} ** 6;

    // Set the hook on our CPU using the global hook function
    address_space.setEcallHook(globalEcallHook);

    // Virtual addresses for our code and stack
    const code_vaddr: u64 = 0x1000;
    const stack_vaddr: u64 = 0x2000;

    // Map memory regions with proper permissions
    try cpu.resources.mapMemory(code_vaddr, 4096, PTE_R | PTE_W | PTE_X | PTE_U, address_space);
    try cpu.resources.mapMemory(stack_vaddr, 4096, PTE_R | PTE_W | PTE_U, address_space);

    // Write instructions to memory (RV64I):
    // addi a7, zero, 42  (a7 = 42)      0x02a00893
    // addi a0, zero, 1   (a0 = 1)       0x00100513
    // addi a1, zero, 2   (a1 = 2)       0x00200593
    // ecall              (syscall)      0x00000073
    // ebreak             (halt)         0x00100073
    try cpu.resources.writeMemory(u32, code_vaddr, 0x02a00893, address_space);
    try cpu.resources.writeMemory(u32, code_vaddr + 4, 0x00100513, address_space);
    try cpu.resources.writeMemory(u32, code_vaddr + 8, 0x00200593, address_space);
    try cpu.resources.writeMemory(u32, code_vaddr + 12, 0x00000073, address_space);
    try cpu.resources.writeMemory(u32, code_vaddr + 16, 0x00100073, address_space);

    // Set up the stack pointer
    address_space.registers[2] = stack_vaddr + 4096 - 16; // sp = x2

    // Set the program counter to our code
    address_space.pc = code_vaddr;

    var steps: usize = 0;
    const max_steps = 8;
    while (steps < max_steps) : (steps += 1) {
        // Step until ebreak
        std.debug.print("Step {d}, PC: 0x{x:016}\n", .{ steps, address_space.pc });
        try cpu.executeInstruction(address_space, null);

        const inst = try cpu.resources.readMemory(u32, address_space.pc, address_space);
        if (inst == 0x00100073) { // ebreak
            break;
        }
    }

    // Check that the ecall hook was called
    std.debug.assert(g_ecall_hook_called);
    std.debug.assert(g_last_syscall_num == 42);
    std.debug.assert(g_last_args[0] == 1 and g_last_args[1] == 2);

    // Check that the return value was correctly set in a0
    std.debug.assert(address_space.registers[10] == 84); // 42 * 2 = 84

    std.debug.print("ECALL hook test passed!\n", .{});
    std.debug.print("  Syscall number: {d}\n", .{g_last_syscall_num});
    std.debug.print("  Arguments: [{d}, {d}, {d}, {d}, {d}, {d}]\n", .{ g_last_args[0], g_last_args[1], g_last_args[2], g_last_args[3], g_last_args[4], g_last_args[5] });
    std.debug.print("  Return value in a0: {d}\n", .{address_space.registers[10]});
}

// Test fork() functionality with copy-on-write
fn testFork() !void {
    const allocator = std.testing.allocator;

    var parent_cpu = try RiscVCpu.init(0, allocator);

    // Create parent address space
    const space = try AddressSpace.init(allocator, .Sv39);

    // Virtual addresses for our shared memory region
    const shared_vaddr: u64 = 0x1000;
    const stack_vaddr: u64 = 0x2000;

    // Map memory regions in parent
    try parent_cpu.resources.mapMemory(shared_vaddr, 4096, PTE_R | PTE_W | PTE_U, space);
    try parent_cpu.resources.mapMemory(stack_vaddr, 4096, PTE_R | PTE_W | PTE_U, space);

    // Write some initial data to shared memory
    const initial_value: u32 = 0xDEADBEEF;
    try parent_cpu.resources.writeMemory(u32, shared_vaddr, initial_value, space);
    try parent_cpu.resources.writeMemory(u32, shared_vaddr + 4, 0x12345678, space);

    std.debug.print("Parent wrote initial values: 0x{x:08}, 0x{x:08}\n", .{ initial_value, 0x12345678 });

    // Verify parent can read the data
    const parent_read1 = try parent_cpu.resources.readMemory(u32, shared_vaddr, space);
    const parent_read2 = try parent_cpu.resources.readMemory(u32, shared_vaddr + 4, space);
    std.debug.assert(parent_read1 == initial_value);
    std.debug.assert(parent_read2 == 0x12345678);

    std.debug.print("Parent verified initial read: 0x{x:08}, 0x{x:08}\n", .{ parent_read1, parent_read2 });

    // Fork the process - this should create a child with CoW pages
    const child_space = try space.clone(parent_cpu.resources.manager);

    std.debug.print("Fork completed successfully\n", .{});

    // Both parent and child should initially see the same data
    const child_read1 = try parent_cpu.resources.readMemory(u32, shared_vaddr, child_space);
    const child_read2 = try parent_cpu.resources.readMemory(u32, shared_vaddr + 4, child_space);
    std.debug.assert(child_read1 == initial_value);
    std.debug.assert(child_read2 == 0x12345678);

    std.debug.print("Child verified initial read: 0x{x:08}, 0x{x:08}\n", .{ child_read1, child_read2 });

    // Now test copy-on-write: parent writes to memory
    const parent_new_value: u32 = 0xCAFEBABE;
    try parent_cpu.resources.writeMemory(u32, shared_vaddr, parent_new_value, space);

    std.debug.print("Parent wrote new value: 0x{x:08}\n", .{parent_new_value});

    // Parent should see the new value
    const parent_after_write = try parent_cpu.resources.readMemory(u32, shared_vaddr, space);
    std.debug.assert(parent_after_write == parent_new_value);

    // Child should still see the original value (CoW should have triggered)
    const child_after_parent_write = try parent_cpu.resources.readMemory(u32, shared_vaddr, child_space);
    std.debug.assert(child_after_parent_write == initial_value);

    std.debug.print("After parent write - Parent: 0x{x:08}, Child: 0x{x:08}\n", .{ parent_after_write, child_after_parent_write });

    // Now child writes to memory
    const child_new_value: u32 = 0xFEEDFACE;
    try parent_cpu.resources.writeMemory(u32, shared_vaddr + 4, child_new_value, child_space);

    std.debug.print("Child wrote new value: 0x{x:08}\n", .{child_new_value});

    // Child should see its new value
    const child_after_write = try parent_cpu.resources.readMemory(u32, shared_vaddr + 4, child_space);
    std.debug.assert(child_after_write == child_new_value);

    // Parent should still see the original value at that location
    const parent_after_child_write = try parent_cpu.resources.readMemory(u32, shared_vaddr + 4, space);
    std.debug.assert(parent_after_child_write == 0x12345678);

    std.debug.print("After child write - Parent: 0x{x:08}, Child: 0x{x:08}\n", .{ parent_after_child_write, child_after_write });

    // Verify both processes have their own independent memory now
    std.debug.print("Final state verification:\n", .{});
    std.debug.print("  Parent addr 0x{x}: 0x{x:08}\n", .{ shared_vaddr, try parent_cpu.resources.readMemory(u32, shared_vaddr, space) });
    std.debug.print("  Parent addr 0x{x}: 0x{x:08}\n", .{ shared_vaddr + 4, try parent_cpu.resources.readMemory(u32, shared_vaddr + 4, space) });
    std.debug.print("  Child addr 0x{x}: 0x{x:08}\n", .{ shared_vaddr, try parent_cpu.resources.readMemory(u32, shared_vaddr, child_space) });
    std.debug.print("  Child addr 0x{x}: 0x{x:08}\n", .{ shared_vaddr + 4, try parent_cpu.resources.readMemory(u32, shared_vaddr + 4, child_space) });

    std.debug.print("Fork test passed! Copy-on-write is working correctly.\n", .{});

    space.deinit(parent_cpu.resources.manager);
    child_space.deinit(parent_cpu.resources.manager);
    parent_cpu.deinit(allocator);
}

// Test fork() with different execution paths (like a real fork scenario)
fn testForkDifferentPaths() !void {
    const allocator = std.testing.allocator;

    var cpu = try RiscVCpu.init(0, allocator);
    defer cpu.deinit(allocator);

    // Create parent address space
    const clone = try AddressSpace.init(allocator, .Sv39);
    defer clone.deinit(cpu.resources.manager);

    // Virtual addresses
    const shared_data_vaddr: u64 = 0x2000;
    const parent_data_vaddr: u64 = 0x3000;
    const child_data_vaddr: u64 = 0x4000;
    const stack_vaddr: u64 = 0x5000;

    // Map memory regions
    try cpu.resources.mapMemory(shared_data_vaddr, 4096, PTE_R | PTE_W | PTE_U, clone);
    try cpu.resources.mapMemory(parent_data_vaddr, 4096, PTE_R | PTE_W | PTE_U, clone);
    try cpu.resources.mapMemory(child_data_vaddr, 4096, PTE_R | PTE_W | PTE_U, clone);
    try cpu.resources.mapMemory(stack_vaddr, 4096, PTE_R | PTE_W | PTE_U, clone);

    // Initialize shared data: [process_id, counter]
    try cpu.resources.writeMemory(u32, shared_data_vaddr, 1, clone); // Parent PID = 1
    try cpu.resources.writeMemory(u32, shared_data_vaddr + 4, 100, clone); // Initial counter = 100

    std.debug.print("Initial shared data: PID={d}, Counter={d}\n", .{ try cpu.resources.readMemory(u32, shared_data_vaddr, clone), try cpu.resources.readMemory(u32, shared_data_vaddr + 4, clone) });

    // Fork the process
    const child_space = try clone.clone(cpu.resources.manager);
    defer child_space.deinit(cpu.resources.manager);

    // Set child PID to 0 (this will trigger CoW when we write)
    try cpu.resources.writeMemory(u32, shared_data_vaddr, 0, child_space);

    std.debug.print("After fork - Parent PID: {d}, Child PID: {d}\n", .{ try cpu.resources.readMemory(u32, shared_data_vaddr, clone), try cpu.resources.readMemory(u32, shared_data_vaddr, child_space) });

    // Simulate parent process behavior
    std.debug.print("Simulating parent process behavior...\n", .{});

    // Parent: increment counter by 10 and write signature
    const parent_old_counter = try cpu.resources.readMemory(u32, shared_data_vaddr + 4, clone);
    try cpu.resources.writeMemory(u32, shared_data_vaddr + 4, parent_old_counter + 10, clone);
    try cpu.resources.writeMemory(u32, parent_data_vaddr, 0xAAA, clone);

    // Simulate child process behavior
    std.debug.print("Simulating child process behavior...\n", .{});

    // Child: increment counter by 5 and write signature
    const child_old_counter = try cpu.resources.readMemory(u32, shared_data_vaddr + 4, child_space);
    try cpu.resources.writeMemory(u32, shared_data_vaddr + 4, child_old_counter + 5, child_space);
    try cpu.resources.writeMemory(u32, child_data_vaddr, 0xBBB, child_space);

    // Check final results
    const parent_counter = try cpu.resources.readMemory(u32, shared_data_vaddr + 4, clone);
    const child_counter = try cpu.resources.readMemory(u32, shared_data_vaddr + 4, child_space);
    const parent_signature = try cpu.resources.readMemory(u32, parent_data_vaddr, clone);
    const child_signature = try cpu.resources.readMemory(u32, child_data_vaddr, child_space);

    std.debug.print("Final results:\n", .{});
    std.debug.print("  Parent counter: {d} (should be 110)\n", .{parent_counter});
    std.debug.print("  Child counter: {d} (should be 105)\n", .{child_counter});
    std.debug.print("  Parent signature: 0x{x:03} (should be 0xAAA)\n", .{parent_signature});
    std.debug.print("  Child signature: 0x{x:03} (should be 0xBBB)\n", .{child_signature});

    // Verify the results - this demonstrates CoW working correctly
    std.debug.assert(parent_counter == 110); // 100 + 10
    std.debug.assert(child_counter == 105); // 100 + 5
    std.debug.assert(parent_signature == 0xAAA);
    std.debug.assert(child_signature == 0xBBB);

    // Verify that parent_data and child_data are independent
    const parent_child_data = try cpu.resources.readMemory(u32, child_data_vaddr, clone);
    const child_parent_data = try cpu.resources.readMemory(u32, parent_data_vaddr, child_space);

    std.debug.print("  Parent sees child_data: 0x{x:03} (should be 0)\n", .{parent_child_data});
    std.debug.print("  Child sees parent_data: 0x{x:03} (should be 0)\n", .{child_parent_data});

    std.debug.assert(parent_child_data == 0); // Parent didn't write to child_data
    std.debug.assert(child_parent_data == 0); // Child didn't write to parent_data

    std.debug.print("Fork different paths test passed!\n", .{});
}

// Test fork() with actual RISC-V code execution
fn testForkWithExecution() !void {
    const allocator = std.testing.allocator;

    var parent_cpu = try RiscVCpu.init(0, allocator);
    defer parent_cpu.deinit(allocator);

    // Create parent address space
    const parent_space = try AddressSpace.init(allocator, .Sv39);
    defer parent_space.deinit(parent_cpu.resources.manager);

    // Virtual addresses
    const code_vaddr: u64 = 0x1000;
    const data_vaddr: u64 = 0x2000;
    const stack_vaddr: u64 = 0x3000;

    // Map memory regions
    try parent_cpu.resources.mapMemory(code_vaddr, 4096, PTE_R | PTE_W | PTE_X | PTE_U, parent_space);
    try parent_cpu.resources.mapMemory(data_vaddr, 4096, PTE_R | PTE_W | PTE_U, parent_space);
    try parent_cpu.resources.mapMemory(stack_vaddr, 4096, PTE_R | PTE_W | PTE_U, parent_space);

    // Write a simple program that increments a value in memory
    // lw   t0, 0(a0)     # Load value from address in a0
    // addi t0, t0, 1     # Increment by 1
    // sw   t0, 0(a0)     # Store back to memory
    // ebreak             # Halt
    try parent_cpu.resources.writeMemory(u32, code_vaddr, 0x00052283, parent_space); // lw t0, 0(a0)
    try parent_cpu.resources.writeMemory(u32, code_vaddr + 4, 0x00128293, parent_space); // addi t0, t0, 1
    try parent_cpu.resources.writeMemory(u32, code_vaddr + 8, 0x00552023, parent_space); // sw t0, 0(a0)
    try parent_cpu.resources.writeMemory(u32, code_vaddr + 12, 0x00100073, parent_space); // ebreak

    // Initialize data value
    const initial_data: u32 = 100;
    try parent_cpu.resources.writeMemory(u32, data_vaddr, initial_data, parent_space);

    // Set up parent CPU state
    parent_space.pc = code_vaddr;
    parent_space.registers[10] = data_vaddr; // a0 = address of data
    parent_space.registers[2] = stack_vaddr + 4096 - 16; // sp

    std.debug.print("Parent initial setup complete. Data value: {d}\n", .{initial_data});

    // Fork the process
    const child_space = try parent_space.clone(parent_cpu.resources.manager);
    defer child_space.deinit(parent_cpu.resources.manager);

    std.debug.print("Fork completed. Both processes should see data value: {d}\n", .{try parent_cpu.resources.readMemory(u32, data_vaddr, child_space)});

    // Execute parent process
    std.debug.print("Executing parent process...\n", .{});
    var parent_steps: usize = 0;
    while (parent_steps < 10) : (parent_steps += 1) {
        try parent_cpu.executeInstruction(parent_space, null);
        const inst = try parent_cpu.resources.readMemory(u32, parent_space.pc, parent_space);
        if (inst == 0x00100073) { // ebreak
            break;
        }
    }

    // Execute child process (reset PC first since it shares the same CPU object)
    std.debug.print("Executing child process...\n", .{});
    child_space.pc = code_vaddr; // Reset PC for child
    child_space.registers[10] = data_vaddr; // a0 = address of data
    var child_steps: usize = 0;
    while (child_steps < 10) : (child_steps += 1) {
        try parent_cpu.executeInstruction(child_space, null);
        const inst = try parent_cpu.resources.readMemory(u32, child_space.pc, child_space);
        if (inst == 0x00100073) { // ebreak
            break;
        }
    }

    // Check final values - they should be different due to CoW
    const parent_final = try parent_cpu.resources.readMemory(u32, data_vaddr, parent_space);
    const child_final = try parent_cpu.resources.readMemory(u32, data_vaddr, child_space);

    std.debug.print("Final values after execution:\n", .{});
    std.debug.print("  Parent: {d}\n", .{parent_final});
    std.debug.print("  Child: {d}\n", .{child_final});

    // Both should have incremented from 100 to 101, but independently
    std.debug.assert(parent_final == 101);
    std.debug.assert(child_final == 101);

    std.debug.print("Fork with execution test passed!\n", .{});
}

test "Testing ECALL hook functionality..." {
    std.debug.print("\nTesting ECALL hook functionality...\n", .{});
    try testEcallHook();
}

test "Testing CPU add shared memory..." {
    std.debug.print("\nTesting CPU add shared memory...\n", .{});
    try testCpuAddSharedMemory();

    std.debug.print("\nAll tests passed!\n", .{});
}

test "Testing fork() functionality..." {
    std.debug.print("\nTesting fork() functionality...\n", .{});
    try testFork();
}

test "Testing fork() with different execution paths..." {
    std.debug.print("\nTesting fork() with different execution paths...\n", .{});
    try testForkDifferentPaths();
}

test "Testing fork() with code execution..." {
    std.debug.print("\nTesting fork() with code execution...\n", .{});
    try testForkWithExecution();
}
