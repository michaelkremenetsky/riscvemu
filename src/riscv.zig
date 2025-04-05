// RISC-V CPU implementation
const std = @import("std");
const copysign = std.math.copysign;

const AccessType = @import("memory.zig").AccessType;
const AddressSpace = @import("memory.zig").AddressSpace;
const AtomicU64 = @import("wasm.zig").AtomicU64;
const print = @import("wasm.zig").print;
const vm_integration = @import("vm_integration.zig");
const Resources = vm_integration.MMUResources;

// Import our virtual memory system
const softfloat = @cImport({
    @cInclude("softfloat.h");
});

const XLEN = 64;
const NUM_REGISTERS = 32;

// Global fence barrier for all threads and fence instructions
var fence_barrier: u32 = 0;

const remove_atomic_align_checks = true;
pub var global_reservation_tracker = GlobalReservationTracker.init();

// Do not pass in unsigned types in here
// std.meta.Int(.unsigned, @typeInfo(sign).int.bits) gets the unsigned type of the sign type
// std.meta.Int(.signed, @typeInfo(T).int.bits) gets the signed type of the T type
fn signExtend(
    comptime T: type,
    comptime sign: type,
    value: anytype,
) T {
    return @bitCast(@as(std.meta.Int(.signed, @typeInfo(T).int.bits), @as(sign, @bitCast(@as(std.meta.Int(.unsigned, @typeInfo(sign).int.bits), @truncate(value))))));
}

// softfloat's fast-int conversions return int_fast32_t, whose width differs
// between targets (i32 on wasm32/darwin, i64 on linux glibc). Pull the real
// return type off the function instead of hardcoding it.
fn ReturnTypeOf(comptime func: anytype) type {
    return @typeInfo(@TypeOf(func)).@"fn".return_type.?;
}

// Notes:
// - Look through the compressed just look at the places with a note.

// Use our MMUResources from vm_integration instead of the original Resources
pub const RiscVCpu = struct {
    // Use MMUResources instead of raw resources
    resources: *Resources,

    // Hart ID for this CPU (used for global reservation tracking)
    hart_id: u32,

    // Initialize with MMUResources
    pub fn init(memory_base: u64, allocator: std.mem.Allocator) !RiscVCpu {
        const resources = try Resources.init(allocator, memory_base);
        const hart_id = global_reservation_tracker.allocateHartId();

        return RiscVCpu{
            .resources = resources,
            .hart_id = hart_id,
        };
    }

    pub fn deinit(self: *RiscVCpu, allocator: std.mem.Allocator) void {
        // Clear any reservations for this hart
        global_reservation_tracker.clearReservation(self.hart_id);
        self.resources.deinit(allocator);
    }

    // Helper function to invalidate LR/SC reservations for atomic operations
    fn invalidateReservation(self: *RiscVCpu) void {
        global_reservation_tracker.clearReservation(self.hart_id);
    }

    // Public method to invalidate reservations (for context switches, interrupts, etc.)
    pub fn invalidateAllReservations(self: *RiscVCpu) void {
        global_reservation_tracker.clearReservation(self.hart_id);
    }

    // Public method to invalidate all reservations across all harts
    pub fn invalidateAllReservationsGlobal() void {
        global_reservation_tracker.clearAllReservations();
    }

    pub fn executeInstruction(self: *RiscVCpu, memory: *AddressSpace, user_data: ?*anyopaque) !void {
        // Lock the CPU state to prevent concurrent access from other threads
        memory.cpu_state_mutex.lock();
        defer memory.cpu_state_mutex.unlock();

        // Normal execution
        const inst = try self.fetchInstruction(memory);
        var pc_was_set = false;

        // std.debug.print("PC: 0x{x:0>16}, Instruction: 0x{x:0>8}\n", .{ address_space.pc, inst });

        // Check if this is a compressed instruction
        if ((inst & 0x3) != 0x3) {
            pc_was_set = try self.executeCompressedInstruction(@truncate(inst), memory);
            if (!pc_was_set) memory.pc += 2;
        } else {
            pc_was_set = try self.executeGeneralInstruction(inst, memory, user_data);
            if (!pc_was_set) memory.pc += 4;
        }

        // Ensure x0 is always 0 (temp fix, probably faster way to do this)
        memory.registers[0] = 0;
    }

    pub fn fetchInstruction(self: *RiscVCpu, memory: *AddressSpace) !u32 {
        // Try to read the instruction through virtual memory system (checking for execute permissions)
        return self.resources.readMemory(u32, memory.pc, memory) catch |err| {
            switch (err) {
                vm_integration.MMUResourcesError.PageFault => {
                    // Handle instruction page fault - we'll allocate memory on demand
                    try self.resources.handlePageFault(memory.pc, AccessType.Execute, memory);
                    // Retry the fetch now that memory is allocated
                    return self.resources.readMemory(u32, memory.pc, memory) catch {
                        return error.InstructionPageFault;
                    };
                },
                else => return error.InstructionAccessFault,
            }
        };
    }

    pub fn executeCompressedInstruction(self: *RiscVCpu, inst: u16, memory: *AddressSpace) !bool {
        const op = inst & 0x3;
        const funct3 = (inst >> 13) & 0x7;

        // std.debug.print("Full instruction: 0x{x:0>8}, lowest 2 bits: 0x{x}\n", .{ inst, inst & 0x3 });

        // Print opcode
        // std.debug.print("Opcode: {d}\n", .{op});
        // std.debug.print("Funct3: {d}\n", .{funct3});

        switch (op) {
            0 => {
                switch (funct3) {
                    0x0 => {
                        // c.addi4spn
                        const rd = ((inst >> 2) & 0x7) + 8;

                        // nzuimm[5:4|9:6|2|3] = inst[12:11|10:7|6|5]
                        const nzuimm = ((inst >> 1) & 0x3c0) // nzuimm[9:6]
                            | ((inst >> 7) & 0x30) // nzuimm[5:4]
                            | ((inst >> 2) & 0x8) // nzuimm[3]
                            | ((inst >> 4) & 0x4); // nzuimm[2]

                        // If this called it usally means there is something wrong with loading the memory.
                        if (nzuimm == 0) {
                            print("If this called it usally means there is something wrong with loading the memory.\n", .{});
                            print("IllegalInstruction: c.addi4spn with nzuimm=0, inst=0x{x}, PC=0x{x}\n", .{ inst, memory.pc });
                            return error.IllegalInstruction;
                        }

                        memory.registers[rd] = memory.registers[2] +% nzuimm;
                    },
                    0x1 => {
                        // c.fld
                        const rd = ((inst >> 2) & 0x7) + 8;
                        const rs1 = ((inst >> 7) & 0x7) + 8;
                        // offset[5:3|7:6] = isnt[12:10|6:5]
                        const offset = ((inst << 1) & 0xc0) // imm[7:6]
                            | ((inst >> 7) & 0x38); // imm[5:3]
                        const val: f64 = @bitCast(
                            try self.resources.readMemory(u64, memory.registers[rs1] +% offset, memory),
                        );
                        memory.fregs[rd] = val;
                    },
                    0x2 => {
                        // c.lw
                        const rd = ((inst >> 2) & 0x7) + 8;
                        const rs1 = ((inst >> 7) & 0x7) + 8;
                        // offset[5:3|2|6] = isnt[12:10|6|5]
                        const offset = ((inst << 1) & 0x40) // imm[6]
                            | ((inst >> 7) & 0x38) // imm[5:3]
                            | ((inst >> 4) & 0x4); // imm[2]
                        const addr = memory.registers[rs1] +% offset;
                        const val = try self.resources.readMemory(u32, addr, memory);
                        memory.registers[rd] = signExtend(u64, i32, val);
                    },
                    0x3 => {
                        // c.ld
                        const rd = ((inst >> 2) & 0x7) + 8;
                        const rs1 = ((inst >> 7) & 0x7) + 8;
                        // offset[5:3|7:6] = isnt[12:10|6:5]
                        const offset = ((inst << 1) & 0xc0) // imm[7:6]
                            | ((inst >> 7) & 0x38); // imm[5:3]
                        const addr = memory.registers[rs1] +% offset;
                        const val = try self.resources.readMemory(u64, addr, memory);
                        memory.registers[rd] = val;
                    },
                    0x4 => {
                        // Reserved.
                        return error.IllegalInstruction;
                    },
                    0x5 => {
                        // c.fsd
                        const rs2 = ((inst >> 2) & 0x7) + 8;
                        const rs1 = ((inst >> 7) & 0x7) + 8;
                        // offset[5:3|7:6] = isnt[12:10|6:5]
                        const offset = ((inst << 1) & 0xc0) // imm[7:6]
                            | ((inst >> 7) & 0x38); // imm[5:3]
                        const addr = memory.registers[rs1] +% offset;
                        try self.resources.writeMemory(u64, addr, @bitCast(memory.fregs[rs2]), memory);
                    },
                    0x6 => {
                        // c.sw
                        const rs2 = ((inst >> 2) & 0x7) + 8;
                        const rs1 = ((inst >> 7) & 0x7) + 8;
                        // offset[5:3|2|6] = isnt[12:10|6|5]
                        const offset = ((inst << 1) & 0x40) // imm[6]
                            | ((inst >> 7) & 0x38) // imm[5:3]
                            | ((inst >> 4) & 0x4); // imm[2]
                        const addr = memory.registers[rs1] +% offset;

                        // NOTE: Not sure about this @truncate
                        try self.resources.writeMemory(u32, addr, @truncate(memory.registers[rs2]), memory);
                    },
                    0x7 => {
                        // c.sd
                        const rs2 = ((inst >> 2) & 0x7) + 8;
                        const rs1 = ((inst >> 7) & 0x7) + 8;
                        // offset[5:3|7:6] = isnt[12:10|6:5]
                        const offset = ((inst << 1) & 0xc0) // imm[7:6]
                            | ((inst >> 7) & 0x38); // imm[5:3]
                        const addr = memory.registers[rs1] +% offset;
                        try self.resources.writeMemory(u64, addr, memory.registers[rs2], memory);
                    },
                    else => return error.UnimplementedInstruction,
                }
            },
            1 => {
                switch (funct3) {
                    0x0 => {
                        // c.addi
                        // Expands to addi rd, rd, nzimm.
                        const rd = (inst >> 7) & 0x1f;
                        // nzimm[5|4:0] = inst[12|6:2]
                        var nzimm: u64 = ((inst >> 7) & 0x20) | ((inst >> 2) & 0x1f);

                        nzimm = if ((nzimm & 0x20) == 0)
                            nzimm
                        else
                            signExtend(u64, i8, 0xc0 | nzimm);

                        if (rd != 0) {
                            memory.registers[rd] = memory.registers[rd] +% nzimm;
                        }
                    },
                    0x1 => {
                        // c.addiw
                        // Expands to addiw rd, rd, imm
                        // "The immediate can be zero for C.ADDIW, where this corresponds to sext.w
                        // rd"
                        const rd = (inst >> 7) & 0x1f;
                        // nzimm[5|4:0] = inst[12|6:2]
                        var imm: u64 = ((inst >> 7) & 0x20) | ((inst >> 2) & 0x1f);

                        imm = if ((imm & 0x20) == 0)
                            imm
                        else
                            signExtend(u64, i8, 0xc0 | imm);

                        if (rd != 0) {
                            memory.registers[rd] = signExtend(u64, i32, memory.registers[rd] +% imm);
                        }
                    },
                    0x2 => {
                        // c.li
                        // Expands to addi rd, x0, imm.
                        const rd = (inst >> 7) & 0x1f;
                        // imm[5|4:0] = inst[12|6:2]
                        var imm: u64 = ((inst >> 7) & 0x20) | ((inst >> 2) & 0x1f);
                        imm = if ((imm & 0x20) == 0)
                            imm
                        else
                            signExtend(u64, i8, 0xc0 | imm);

                        if (rd != 0) {
                            memory.registers[rd] = imm;
                        }
                    },
                    0x3 => {
                        const rd = (inst >> 7) & 0x1f;

                        switch (rd) {
                            0 => {
                                // Reserved
                            },
                            2 => {
                                // c.addi16sp
                                // Expands to addi x2, x2, nzimm
                                // nzimm[9|4|6|8:7|5] = inst[12|6|5|4:3|2]
                                var nzimm: u64 = ((inst >> 3) & 0x200) // nzimm[9]
                                    | ((inst >> 2) & 0x10) // nzimm[4]
                                    | ((inst << 1) & 0x40) // nzimm[6]
                                    | ((inst << 4) & 0x180) // nzimm[8:7]
                                    | ((inst << 3) & 0x20); // nzimm[5]

                                nzimm = if ((nzimm & 0x200) == 0)
                                    nzimm
                                else
                                    @as(u64, @bitCast(@as(i64, @as(i32, @as(i16, @bitCast(@as(u16, @intCast(0xfc00 | nzimm))))))));

                                if (nzimm != 0) {
                                    memory.registers[2] = memory.registers[2] +% nzimm;
                                }
                            },
                            else => {
                                // c.lui
                                // Expands to lui rd, nzimm.
                                // nzimm[17|16:12] = inst[12|6:2]

                                // NOTE: Remove the @as(u64, inst)?
                                var nzimm: u64 = ((@as(u64, inst) << 5) & 0x20000) | ((@as(u64, inst) << 10) & 0x1f000);

                                nzimm = if ((nzimm & 0x20000) == 0)
                                    nzimm
                                else
                                    signExtend(u64, i32, 0xfffc0000 | nzimm);

                                if (nzimm != 0) {
                                    memory.registers[rd] = nzimm;
                                }
                            },
                        }
                    },
                    0x4 => {
                        const funct2 = (inst >> 10) & 0x3;
                        switch (funct2) {
                            0x0 => { // c.srli
                                const rd = ((inst >> 7) & 0b111) + 8;
                                // shamt[5|4:0] = inst[12|6:2]
                                const shamt = ((inst >> 7) & 0x20) | ((inst >> 2) & 0x1f);
                                memory.registers[rd] = memory.registers[rd] >> @intCast(shamt);
                            },
                            0x1 => { // c.srai
                                const rd = ((inst >> 7) & 0b111) + 8;
                                // shamt[5|4:0] = inst[12|6:2]
                                const shamt = ((inst >> 7) & 0x20) | ((inst >> 2) & 0x1f);
                                memory.registers[rd] = @as(u64, @bitCast(@as(i64, @bitCast(memory.registers[rd])) >> @intCast(shamt)));
                            },
                            0x2 => { // c.andi
                                const rd = ((inst >> 7) & 0b111) + 8;
                                // imm[5|4:0] = inst[12|6:2]
                                var imm: u64 = ((inst >> 7) & 0x20) | ((inst >> 2) & 0x1f);
                                // Sign-extended
                                imm = if ((imm & 0x20) == 0)
                                    imm
                                else
                                    signExtend(u64, i8, 0xc0 | imm);

                                memory.registers[rd] = memory.registers[rd] & imm;
                            },
                            // NOTE: Not sure if this is correct. Seems right?
                            0x3 => {
                                const bit12 = (inst >> 12) & 0b1;
                                const bit5_6 = (inst >> 5) & 0b11;
                                switch (bit12) {
                                    0x0 => switch (bit5_6) {
                                        0x0 => { // c.sub
                                            const rd = ((inst >> 7) & 0b111) + 8;
                                            const rs2 = ((inst >> 2) & 0b111) + 8;
                                            memory.registers[rd] = memory.registers[rd] -% memory.registers[rs2];
                                        },
                                        0x1 => { // c.xor
                                            const rd = ((inst >> 7) & 0b111) + 8;
                                            const rs2 = ((inst >> 2) & 0b111) + 8;
                                            memory.registers[rd] = memory.registers[rd] ^ memory.registers[rs2];
                                        },
                                        0x2 => { // c.or
                                            const rd = ((inst >> 7) & 0b111) + 8;
                                            const rs2 = ((inst >> 2) & 0b111) + 8;
                                            memory.registers[rd] = memory.registers[rd] | memory.registers[rs2];
                                        },
                                        0x3 => { // c.and
                                            const rd = ((inst >> 7) & 0b111) + 8;
                                            const rs2 = ((inst >> 2) & 0b111) + 8;
                                            memory.registers[rd] = memory.registers[rd] & memory.registers[rs2];
                                        },
                                        else => return error.IllegalInstruction,
                                    },
                                    0x1 => switch (bit5_6) {
                                        0x0 => { // c.subw
                                            const rd = ((inst >> 7) & 0b111) + 8;
                                            const rs2 = ((inst >> 2) & 0b111) + 8;

                                            memory.registers[rd] = signExtend(u64, i32, memory.registers[rd] -% memory.registers[rs2]);
                                        },
                                        0x1 => { // c.addw
                                            const rd = ((inst >> 7) & 0b111) + 8;
                                            const rs2 = ((inst >> 2) & 0b111) + 8;
                                            memory.registers[rd] = signExtend(u64, i32, memory.registers[rd] +% memory.registers[rs2]);
                                        },
                                        else => return error.IllegalInstruction,
                                    },
                                    else => return error.IllegalInstruction,
                                }
                            },
                            else => return error.IllegalInstruction,
                        }
                    },
                    0x5 => {
                        // c.j
                        // Expands to jal x0, offset.
                        var offset: u64 = ((inst >> 1) & 0x800) // offset[11]
                            | ((inst << 2) & 0x400) // offset[10]
                            | ((inst >> 1) & 0x300) // offset[9:8]
                            | ((inst << 1) & 0x80) // offset[7]
                            | ((inst >> 1) & 0x40) // offset[6]
                            | ((inst << 3) & 0x20) // offset[5]
                            | ((inst >> 7) & 0x10) // offset[4]
                            | ((inst >> 2) & 0xe); // offset[3:1]

                        offset = if ((offset & 0x800) == 0)
                            offset
                        else
                            signExtend(u64, i16, 0xf000 | offset);

                        memory.pc = memory.pc +% offset;
                        return true;
                    },
                    0x6 => {
                        // c.beqz
                        // Expands to beq rs1, x0, offset, rs1=rs1'+8.
                        const rs1 = ((inst >> 7) & 0b111) + 8;

                        var offset: u64 = ((inst >> 4) & 0x100) // offset[8]
                            | ((inst << 1) & 0xc0) // offset[7:6]
                            | ((inst << 3) & 0x20) // offset[5]
                            | ((inst >> 7) & 0x18) // offset[4:3]
                            | ((inst >> 2) & 0x6); // offset[2:1]

                        offset = if ((offset & 0x100) == 0)
                            offset
                        else
                            signExtend(u64, i16, 0xfe00 | offset);

                        if (memory.registers[rs1] == 0) {
                            memory.pc = memory.pc +% offset;
                            return true;
                        }
                    },
                    0x7 => {
                        // c.bnez
                        // Expands to bne rs1, x0, offset, rs1=rs1'+8.
                        const rs1 = ((inst >> 7) & 0b111) + 8;

                        var offset: u64 = ((inst >> 4) & 0x100) // offset[8]
                            | ((inst << 1) & 0xc0) // offset[7:6]
                            | ((inst << 3) & 0x20) // offset[5]
                            | ((inst >> 7) & 0x18) // offset[4:3]
                            | ((inst >> 2) & 0x6); // offset[2:1]

                        offset = if ((offset & 0x100) == 0)
                            offset
                        else
                            signExtend(u64, i16, 0xfe00 | offset);

                        if (memory.registers[rs1] != 0) {
                            memory.pc = memory.pc +% offset;
                            return true;
                        }
                    },
                    else => return error.UnimplementedInstruction,
                }
            },
            2 => {
                switch (funct3) {
                    0x0 => {
                        // c.slli
                        // Expands to slli rd, rd, shamt.
                        const rd = (inst >> 7) & 0x1f;
                        // shamt[5|4:0] = inst[12|6:2]
                        const shamt = ((inst >> 7) & 0x20) | ((inst >> 2) & 0x1f);
                        if (rd != 0) {
                            memory.registers[rd] = memory.registers[rd] << @intCast(shamt);
                        }
                    },
                    0x1 => {
                        // c.fldsp
                        // Expands to fld rd, offset(x2).
                        const rd = (inst >> 7) & 0x1f;
                        // offset[5|4:3|8:6] = inst[12|6:5|4:2]
                        const offset = ((inst << 4) & 0x1c0) // offset[8:6]
                            | ((inst >> 7) & 0x20) // offset[5]
                            | ((inst >> 2) & 0x18); // offset[4:3]
                        memory.fregs[rd] = @bitCast(try self.resources.readMemory(u64, memory.registers[2] +% offset, memory));
                    },
                    0x2 => {
                        // c.lwsp
                        // Expands to lw rd, offset(x2).
                        const rd = (inst >> 7) & 0x1f;
                        // offset[5|4:2|7:6] = inst[12|6:4|3:2]
                        const offset = ((inst << 4) & 0xc0) // offset[7:6]
                            | ((inst >> 7) & 0x20) // offset[5]
                            | ((inst >> 2) & 0x1c); // offset[4:2]
                        const val = try self.resources.readMemory(u32, memory.registers[2] +% offset, memory);
                        memory.registers[rd] = signExtend(u64, i32, val);
                    },
                    0x3 => {
                        // c.ldsp
                        // Expands to ld rd, offset(x2).
                        const rd = (inst >> 7) & 0x1f;
                        // offset[5|4:3|8:6] = inst[12|6:5|4:2]
                        const offset = ((inst << 4) & 0x1c0) // offset[8:6]
                            | ((inst >> 7) & 0x20) // offset[5]
                            | ((inst >> 2) & 0x18); // offset[4:3]
                        const val = try self.resources.readMemory(u64, memory.registers[2] +% offset, memory);
                        memory.registers[rd] = val;
                    },
                    // NOTE: Not sure if this is correct. Look at this again
                    0x4 => {
                        const bit12 = (inst >> 12) & 0x1;
                        const rs2 = (inst >> 2) & 0x1f;
                        if (bit12 == 0) {
                            if (rs2 == 0) {
                                // c.jr
                                // Expands to jalr x0, 0(rs1).
                                const rs1 = (inst >> 7) & 0x1f;
                                if (rs1 != 0) {
                                    memory.pc = memory.registers[rs1] & ~@as(u64, 1);
                                    return true;
                                }
                            } else {
                                // c.mv
                                // Expands to add rd, x0, rs2.
                                const rd = (inst >> 7) & 0x1f;
                                if (rs2 != 0) {
                                    memory.registers[rd] = memory.registers[rs2];
                                }
                            }
                        } else {
                            const rd = (inst >> 7) & 0x1f;
                            if (rs2 == 0) {
                                if (rd == 0) {
                                    // c.ebreak
                                    // Expands to ebreak.
                                    return error.BreakpointCompressed;
                                } else {
                                    // c.jalr
                                    // Expands to jalr x1, 0(rs1).
                                    const rs1 = (inst >> 7) & 0x1f;
                                    const t = memory.pc +% 2;
                                    memory.pc = memory.registers[rs1] & ~@as(u64, 1); // ~1 not -% 2
                                    memory.registers[1] = t;
                                    return true;
                                }
                            } else {
                                // c.add
                                // Expands to add rd, rd, rs2.
                                if (rs2 != 0) {
                                    memory.registers[rd] = memory.registers[rd] +% memory.registers[rs2];
                                }
                            }
                        }
                    },
                    0x5 => {
                        // c.fsdsp
                        // Expands to fsd rs2, offset(x2).
                        const rs2 = ((inst >> 2) & 0x1f);
                        // offset[5:3|8:6] = isnt[12:10|9:7]
                        const offset = ((inst >> 1) & 0x1c0) // offset[8:6]
                            | ((inst >> 7) & 0x38); // offset[5:3]
                        try self.resources.writeMemory(u64, memory.registers[2] +% offset, @bitCast(memory.fregs[rs2]), memory);
                    },
                    0x6 => {
                        // c.swsp
                        // Expands to sw rs2, offset(x2).
                        const rs2 = ((inst >> 2) & 0x1f);
                        // offset[5:2|7:6] = inst[12:9|8:7]
                        const offset = ((inst >> 1) & 0xc0) // offset[7:6]
                            | ((inst >> 7) & 0x3c); // offset[5:2]
                        const addr = memory.registers[2] +% offset;

                        // NOTE: This causes a panic if we use @intCast()?
                        try self.resources.writeMemory(u32, addr, @truncate(memory.registers[rs2]), memory);
                    },
                    0x7 => {
                        // c.sdsp
                        // Expands to sd rs2, offset(x2).
                        const rs2 = ((inst >> 2) & 0x1f);
                        // offset[5:3|8:6] = isnt[12:10|9:7]
                        const offset = ((inst >> 1) & 0x1c0) // offset[8:6]
                            | ((inst >> 7) & 0x38); // offset[5:3]
                        const addr = memory.registers[2] +% offset;
                        try self.resources.writeMemory(u64, addr, memory.registers[rs2], memory);
                    },
                    else => return error.UnimplementedInstruction,
                }
            },
            else => return error.UnimplementedInstruction,
        }
        return false;
    }

    pub fn executeGeneralInstruction(self: *RiscVCpu, inst: u32, memory: *AddressSpace, user_data: ?*anyopaque) !bool {
        const opcode = inst & 0x7F;
        const rd = (inst >> 7) & 0x1F;
        const rs1 = (inst >> 15) & 0x1F;
        const rs2 = (inst >> 20) & 0x1F;
        const funct3 = (inst >> 12) & 0x7;

        switch (opcode) {
            0x03 => { // LOAD (I-type)
                // RV32I and RV64I
                // imm[11:0] = inst[31:20]
                const offset = @as(u64, @bitCast(@as(i64, @as(i32, @bitCast(inst))) >> 20));
                const addr = memory.registers[rs1] +% offset;

                switch (funct3) {
                    0x0 => { // lb
                        const value = try self.resources.readMemory(u8, addr, memory);
                        memory.registers[rd] = signExtend(u64, i8, value);
                    },
                    0x1 => { // lh
                        const value = try self.resources.readMemory(u16, addr, memory);
                        memory.registers[rd] = signExtend(u64, i16, value);
                    },
                    0x2 => { // lw
                        const value = try self.resources.readMemory(u32, addr, memory);
                        memory.registers[rd] = signExtend(u64, i32, value);
                    },
                    0x3 => { // ld
                        memory.registers[rd] = try self.resources.readMemory(u64, addr, memory);
                    },
                    0x4 => { // lbu
                        memory.registers[rd] = try self.resources.readMemory(u8, addr, memory);
                    },
                    0x5 => { // lhu
                        memory.registers[rd] = try self.resources.readMemory(u16, addr, memory);
                    },
                    0x6 => { // lwu
                        memory.registers[rd] = try self.resources.readMemory(u32, addr, memory);
                    },
                    else => return error.UnimplementedInstruction,
                }
            },
            // NOTE: Not sure about the truncate here
            0x07 => { // LOAD (I-type, F extension)]
                // RV32D and RV64D
                // imm[11:0] = inst[31:20]
                const offset: u64 = @bitCast(@as(i64, @as(i32, @bitCast(inst))) >> 20);
                const addr = memory.registers[rs1] +% offset;

                // TODO: Not sure if this is correct.
                switch (funct3) {
                    0x2 => { // flw
                        const word: u32 = try self.resources.readMemory(u32, addr, memory);
                        // NaN-box the f32 value into the f64 register
                        const nan_boxed_bits = (@as(u64, 0xFFFFFFFF) << 32) | @as(u64, word);
                        memory.fregs[rd] = @as(f64, @bitCast(nan_boxed_bits));
                    },
                    0x3 => { // fld
                        memory.fregs[rd] = @as(f64, @bitCast(try self.resources.readMemory(u64, addr, memory)));
                    },
                    else => return error.UnimplementedInstruction,
                }
            },
            0x0f => { // Fence instructions
                // RV32I and RV64I
                switch (funct3) {
                    0x0 => { // fence
                        // Extract predecessor and successor flags
                        // RISC-V fence instruction format:
                        // inst[31:28] = fm (fence mode)
                        // inst[27:24] = pred (predecessor)
                        // inst[23:20] = succ (successor)
                        const pred = (inst >> 24) & 0xF; // inst[27:24]
                        const succ = (inst >> 20) & 0xF; // inst[23:20]

                        const W = 0x8; // Write
                        const R = 0x4; // Read

                        // Implement different barrier types based on pred/succ flags
                        if (pred == 0 and succ == 0) {
                            // fence none,none - no-op
                            return false;
                        } else if ((pred & (W | R)) != 0 and (succ & (W | R)) != 0) {
                            // Any memory-to-memory fence - full barrier
                            _ = @atomicRmw(u32, &fence_barrier, .Add, 0, .seq_cst);
                        } else if ((pred & (W | R)) != 0) {
                            // Memory to I/O - release semantics
                            @atomicStore(u32, &fence_barrier, 0, .release);
                        } else if ((succ & (W | R)) != 0) {
                            // I/O to memory - acquire semantics
                            _ = @atomicLoad(u32, &fence_barrier, .acquire);
                        } else {
                            // I/O to I/O - lighter barrier
                            _ = @atomicRmw(u32, &fence_barrier, .Add, 0, .acq_rel);
                        }
                    },
                    0x1 => { // fence.i
                        // Instruction fence - ensures all previous instructions are visible before fetching new instructions. This is important for self-modifying code

                        // In a real CPU this would also flush instruction cache
                        // For our emulator, a memory fence is sufficient since we don't have separate instruction/data caches

                        // Use sequential consistency to ensure strongest ordering
                        // This guarantees that all previous memory operations
                        // (including stores that might modify instructions)
                        // are complete before any subsequent instruction fetches
                        _ = @atomicRmw(u32, &fence_barrier, .Add, 0, .seq_cst);
                    },
                    else => return error.UnimplementedInstruction,
                }
            },
            0x13 => {
                // RV32I and RV64I
                // imm[11:0] = inst[31:20]
                const imm = @as(u64, @bitCast(@as(i64, @as(i32, @bitCast(inst))) >> 20));

                const funct7 = (inst >> 25) & 0x7F;
                const funct6 = funct7 >> 1;

                switch (funct3) {
                    0x0 => { // addi (Add Immediate)
                        memory.registers[rd] = memory.registers[rs1] +% imm;
                    },
                    0x1 => { // slli (Shift Left Logical Immediate)
                        // shamt size is 5 bits for RV32I and 6 bits for RV64I.
                        const shamt: u6 = @intCast((inst >> 20) & 0x3f);
                        memory.registers[rd] = memory.registers[rs1] << shamt;
                    },
                    0x2 => { // slti (Set Less Than Immediate)
                        memory.registers[rd] = @intFromBool(@as(i64, @bitCast(memory.registers[rs1])) < @as(i64, @bitCast(imm)));
                    },
                    0x3 => { // sltiu (Set Less Than Immediate Unsigned)
                        memory.registers[rd] = @intFromBool(memory.registers[rs1] < imm);
                    },
                    0x4 => { // xori (XOR Immediate)
                        memory.registers[rd] = memory.registers[rs1] ^ imm;
                    },
                    0x5 => {
                        switch (funct6) {
                            0x00 => { // srli (Shift Right Logical Immediate)
                                // shamt size is 5 bits for RV32I and 6 bits for RV64I.
                                const shamt: u6 = @intCast((inst >> 20) & 0x3f);
                                memory.registers[rd] = memory.registers[rs1] >> shamt;
                            },
                            0x10 => { // srai (Shift Right Arithmetic Immediate)
                                // shamt size is 5 bits for RV32I and 6 bits for RV64I.
                                const shamt: u6 = @intCast((inst >> 20) & 0x3f);
                                memory.registers[rd] = @as(u64, @bitCast(@as(i64, @bitCast(memory.registers[rs1])) >> shamt));
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    0x6 => { // ori (OR Immediate)
                        memory.registers[rd] = memory.registers[rs1] | imm;
                    },
                    0x7 => { // andi (AND Immediate)
                        memory.registers[rd] = memory.registers[rs1] & imm;
                    },
                    else => return error.UnimplementedInstruction,
                }
            },
            0x17 => { // auipc (Add Upper Immediate to PC)
                // RV32I
                // Add Upper Immediate to PC - Forms a 32-bit offset from the 20-bit U-immediate, filling in the lowest 12 bits with zeros.
                // AUIPC enables code to work regardless of where it's loaded in memory by calculating addresses relative to the current PC.
                // It is meant to be used in combination with JALR to reach a 32-bit PC-relative address.

                // imm[31:12] = inst[31:12]
                const imm = signExtend(u64, i32, inst & 0xfffff000);
                memory.registers[rd] = memory.pc +% imm;
            },
            0x1b => {
                // RV64I
                // imm[11:0] = inst[31:20]
                const imm = @as(u64, @bitCast(@as(i64, @as(i32, @bitCast(inst))) >> 20));
                const funct7 = (inst >> 25) & 0x7F;

                switch (funct3) {
                    0x0 => { // addiw (add immediate word)
                        memory.registers[rd] = signExtend(u64, i32, memory.registers[rs1] +% imm);
                    },
                    // NOTE: Not 100% sure about this one
                    0x1 => { // slliw (shift left logical immediate word)
                        // Shift left the lower 32 bits of rs1 by shamt (0-31) bits, then sign-extend to 64 bits.
                        const shamt: u5 = @intCast(imm & 0x1f);
                        const rs1_val: i32 = @bitCast(@as(u32, @truncate(memory.registers[rs1])));
                        const shifted: i32 = rs1_val << shamt;
                        memory.registers[rd] = @bitCast(@as(i64, shifted));
                    },
                    0x5 => {
                        switch (funct7) {
                            0x00 => { // srliw (Shift Right Logical Immediate Word)
                                const shamt: u5 = @intCast(imm & 0x1f);
                                const shifted: u32 = @as(u32, @truncate(memory.registers[rs1])) >> shamt;
                                memory.registers[rd] = signExtend(u64, i32, shifted);
                            },
                            0x20 => { // sraiw (Shift Right Arithmetic Immediate Word)
                                const shamt: u5 = @intCast(imm & 0x1f);
                                const shifted: i32 = @as(i32, @bitCast(@as(u32, @truncate(memory.registers[rs1])))) >> shamt;
                                memory.registers[rd] = @bitCast(@as(i64, shifted));
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    else => return error.UnimplementedInstruction,
                }
            },
            // TODO: Note sure about the @truncates here. Also need to store it as the specfic type of byte?
            0x23 => { // STORE (S-type)
                // RV32I
                // offset[11:5|4:0] = inst[31:25|11:7]
                const offset = @as(u64, @bitCast((@as(i64, @as(i32, (@bitCast((inst & 0xfe000000)))) >> 20)))) | ((inst >> 7) & 0x1f);
                const addr = memory.registers[rs1] +% offset;

                switch (funct3) {
                    0x0 => try self.resources.writeMemory(u8, addr, @truncate(memory.registers[rs2]), memory), // SB (store byte)
                    0x1 => try self.resources.writeMemory(u16, addr, @truncate(memory.registers[rs2]), memory), // SH (store halfword)
                    0x2 => try self.resources.writeMemory(u32, addr, @truncate(memory.registers[rs2]), memory), // SW (store word)
                    0x3 => try self.resources.writeMemory(u64, addr, memory.registers[rs2], memory), // SD (store doubleword)
                    else => return error.UnimplementedInstruction,
                }
            },
            0x27 => {
                // RV32F and RV64F
                // Reconstruct the 12-bit S-type immediate and sign-extend it properly.
                // imm[11:5] = inst[31:25], imm[4:0] = inst[11:7]
                const imm12: u32 = (((inst >> 25) & 0x7F) << 5) | ((inst >> 7) & 0x1F);
                // Sign-extend the 12-bit immediate to 64 bits
                const offset = @as(u64, @bitCast(@as(i64, @as(i32, @bitCast(imm12)) << 20 >> 20)));
                const addr = memory.registers[rs1] +% offset;

                switch (funct3) {
                    0x2 => { // fsw (store word floating-point)
                        // Get the raw 64 bits from the register
                        const f64_bits = @as(u64, @bitCast(memory.fregs[rs2]));
                        // Truncate to get the lower 32 bits
                        const lower_32_bits = @as(u32, @truncate(f64_bits));
                        // Write the 32 bits to memory
                        try self.resources.writeMemory(u32, addr, lower_32_bits, memory);
                    },
                    0x3 => { // fsd (store double-word floating-point)
                        const value: f64 = memory.fregs[rs2];
                        try self.resources.writeMemory(f64, addr, value, memory);
                    },
                    else => return error.UnimplementedInstruction,
                }
            },
            0x2f => {
                // RV32A and RV64A
                const funct5 = (inst >> 27) & 0x1f;

                // Extract acquire and release bits
                const aq = ((inst >> 26) & 0x1) != 0; // Acquire bit
                const rl = ((inst >> 25) & 0x1) != 0; // Release bit

                // Notes for me:
                // - LR (Load-Reserved) and SC (Store-Conditional) instructions should generally be placed as close together as possible in RISC-V code as they are invalidated in many places.

                switch (funct3) {
                    0x2 => {
                        // Check alignment for 32-bit atomics
                        const addr = memory.registers[rs1];

                        if (comptime !remove_atomic_align_checks) {
                            if ((addr & 0x3) != 0) {
                                return error.AddressMisaligned;
                            }
                        }

                        switch (funct5) {
                            0x00 => {
                                // amoadd.w (atomic add word)
                                const value: u32 = @truncate(memory.registers[rs2]);
                                self.invalidateReservation();
                                const old_value = try self.resources.atomicRmwU32(.Add, addr, value, aq, rl, memory);
                                memory.registers[rd] = signExtend(u64, i32, old_value);
                            },
                            0x01 => {
                                // amoswap.w (atomic swap word)
                                const value: u32 = @truncate(memory.registers[rs2]);
                                self.invalidateReservation();
                                const old_value = try self.resources.atomicRmwU32(.Xchg, addr, value, aq, rl, memory);
                                memory.registers[rd] = signExtend(u64, i32, old_value);
                            },
                            0x02 => {
                                // lr.w (load reserved word)
                                const old_value = try self.resources.atomicReadU32(addr, memory);
                                global_reservation_tracker.setReservation(self.hart_id, addr); // Track globally
                                memory.registers[rd] = signExtend(u64, i32, old_value);
                            },
                            0x03 => {
                                // sc.w (store conditional word)
                                const value: u32 = @truncate(memory.registers[rs2]);
                                // Check if we have a valid reservation for this address (both local and global)
                                if (global_reservation_tracker.hasReservation(self.hart_id, addr)) {
                                    try self.resources.writeMemory(u32, addr, value, memory);
                                    memory.registers[rd] = 0; // Success
                                } else {
                                    // No valid reservation
                                    memory.registers[rd] = 1; // Failure
                                }
                                // Clear the reservation in any case
                                global_reservation_tracker.clearReservation(self.hart_id);
                            },
                            0x04 => {
                                // amoxor.w (atomic XOR word)
                                const value: u32 = @truncate(memory.registers[rs2]);
                                self.invalidateReservation();
                                const old_value = try self.resources.atomicRmwU32(.Xor, addr, value, aq, rl, memory);
                                memory.registers[rd] = signExtend(u64, i32, old_value);
                            },
                            0x08 => {
                                // amoor.w (atomic OR word)
                                const value: u32 = @truncate(memory.registers[rs2]);
                                self.invalidateReservation();
                                const old_value = try self.resources.atomicRmwU32(.Or, addr, value, aq, rl, memory);
                                memory.registers[rd] = signExtend(u64, i32, old_value);
                            },
                            0x0c => {
                                // amoand.w (atomic AND word)
                                const value: u32 = @truncate(memory.registers[rs2]);
                                self.invalidateReservation();
                                const old_value = try self.resources.atomicRmwU32(.And, addr, value, aq, rl, memory);
                                memory.registers[rd] = signExtend(u64, i32, old_value);
                            },
                            0x10 => {
                                // amomin.w (atomic minimum word - signed)
                                const value: u32 = @truncate(memory.registers[rs2]);
                                self.invalidateReservation();
                                const old_value = try self.resources.atomicRmwU32(.Min, addr, value, aq, rl, memory);
                                memory.registers[rd] = signExtend(u64, i32, old_value);
                            },
                            0x14 => {
                                // amomax.w (atomic maximum word - signed)
                                const value: u32 = @truncate(memory.registers[rs2]);
                                self.invalidateReservation();
                                const old_value = try self.resources.atomicRmwU32(.Max, addr, value, aq, rl, memory);
                                memory.registers[rd] = signExtend(u64, i32, old_value);
                            },
                            0x18 => {
                                // amominu.w (atomic minimum unsigned word)
                                const value: u32 = @truncate(memory.registers[rs2]);
                                self.invalidateReservation();
                                const old_value = try self.resources.atomicRmwU32Unsigned(.Min, addr, value, aq, rl, memory);
                                memory.registers[rd] = signExtend(u64, i32, old_value);
                            },
                            0x1c => {
                                // amomaxu.w (atomic maximum unsigned word)
                                const value: u32 = @truncate(memory.registers[rs2]);
                                self.invalidateReservation();
                                const old_value = try self.resources.atomicRmwU32Unsigned(.Max, addr, value, aq, rl, memory);
                                memory.registers[rd] = signExtend(u64, i32, old_value);
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    0x3 => {
                        // Check alignment for 64-bit atomics
                        const addr = memory.registers[rs1];

                        if (comptime !remove_atomic_align_checks) {
                            if ((addr & 0x7) != 0) {
                                return error.AddressMisaligned;
                            }
                        }

                        switch (funct5) {
                            0x00 => {
                                // amoadd.d (atomic add doubleword)
                                const value = memory.registers[rs2];
                                self.invalidateReservation();
                                const old_value = try self.resources.atomicRmwU64(.Add, addr, value, aq, rl, memory);
                                memory.registers[rd] = old_value;
                            },
                            0x01 => {
                                // amoswap.d (atomic swap doubleword)
                                const value = memory.registers[rs2];
                                self.invalidateReservation();
                                const old_value = try self.resources.atomicRmwU64(.Xchg, addr, value, aq, rl, memory);
                                memory.registers[rd] = old_value;
                            },
                            0x02 => {
                                // lr.d (load reserved doubleword)
                                const old_value = try self.resources.atomicReadU64(addr, memory);
                                global_reservation_tracker.setReservation(self.hart_id, addr); // Track globally
                                memory.registers[rd] = old_value;
                            },
                            0x03 => {
                                // sc.d (store conditional doubleword)
                                const value = memory.registers[rs2];
                                // Check if we have a valid reservation for this address (both local and global)
                                if (global_reservation_tracker.hasReservation(self.hart_id, addr)) {
                                    try self.resources.writeMemory(u64, addr, value, memory);
                                    memory.registers[rd] = 0; // Success
                                } else {
                                    // No valid reservation
                                    memory.registers[rd] = 1; // Failure
                                }
                                // Clear the reservation in any case
                                global_reservation_tracker.clearReservation(self.hart_id);
                            },
                            0x04 => {
                                // amoxor.d (atomic XOR doubleword)
                                const value = memory.registers[rs2];
                                self.invalidateReservation();
                                const old_value = try self.resources.atomicRmwU64(.Xor, addr, value, aq, rl, memory);
                                memory.registers[rd] = old_value;
                            },
                            0x08 => {
                                // amoor.d (atomic OR doubleword)
                                const value = memory.registers[rs2];
                                self.invalidateReservation();
                                const old_value = try self.resources.atomicRmwU64(.Or, addr, value, aq, rl, memory);
                                memory.registers[rd] = old_value;
                            },
                            0x0c => {
                                // amoand.d (atomic AND doubleword)
                                const value = memory.registers[rs2];
                                self.invalidateReservation();
                                const old_value = try self.resources.atomicRmwU64(.And, addr, value, aq, rl, memory);
                                memory.registers[rd] = old_value;
                            },
                            0x10 => {
                                // amomin.d (atomic minimum doubleword - signed)
                                const value = memory.registers[rs2];
                                self.invalidateReservation();
                                const old_value = try self.resources.atomicRmwU64(.Min, addr, value, aq, rl, memory);
                                memory.registers[rd] = old_value;
                            },
                            0x14 => {
                                // amomax.d (atomic maximum doubleword - signed)
                                const value = memory.registers[rs2];
                                self.invalidateReservation();
                                const old_value = try self.resources.atomicRmwU64(.Max, addr, value, aq, rl, memory);
                                memory.registers[rd] = old_value;
                            },
                            0x18 => {
                                // amominu.d (atomic minimum unsigned doubleword)
                                const value = memory.registers[rs2];
                                self.invalidateReservation();
                                const old_value = try self.resources.atomicRmwU64Unsigned(.Min, addr, value, aq, rl, memory);
                                memory.registers[rd] = old_value;
                            },
                            0x1c => {
                                // amomaxu.d (atomic maximum unsigned doubleword)
                                const value = memory.registers[rs2];
                                self.invalidateReservation();
                                const old_value = try self.resources.atomicRmwU64Unsigned(.Max, addr, value, aq, rl, memory);
                                memory.registers[rd] = old_value;
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    else => return error.UnimplementedInstruction,
                }
            },
            0x33 => {
                // RV64I and RV64M
                const funct7 = (inst >> 25) & 0x7F;

                switch (funct3) {
                    0x0 => {
                        switch (funct7) {
                            0x00 => { // add
                                memory.registers[rd] = memory.registers[rs1] +% memory.registers[rs2];
                            },
                            0x01 => {
                                // mul (multiply)
                                memory.registers[rd] = @bitCast(@as(i64, @bitCast(memory.registers[rs1])) *% @as(i64, @bitCast(memory.registers[rs2])));
                            },
                            0x20 => {
                                // sub (subtract)
                                memory.registers[rd] = memory.registers[rs1] -% memory.registers[rs2];
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    0x1 => {
                        switch (funct7) {
                            0x00 => {
                                // sll (shift left logical)
                                const shamt: u6 = @intCast(memory.registers[rs2] & 0x3f);
                                memory.registers[rd] = memory.registers[rs1] << shamt;
                            },
                            // NOTE: This is supposed to get up to i128 which I didn't add
                            0x01 => {
                                // mulh (multiply high)
                                // Multiplies two signed 64-bit values and returns the upper 64 bits of the 128-bit result
                                const a_i64: i64 = @bitCast(memory.registers[rs1]);
                                const b_i64: i64 = @bitCast(memory.registers[rs2]);
                                const result: i128 = @as(i128, a_i64) * @as(i128, b_i64);
                                // Take the high 64 bits of the 128-bit result
                                memory.registers[rd] = @bitCast(@as(i64, @truncate(result >> 64)));
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    0x2 => {
                        switch (funct7) {
                            0x00 => {
                                // slt (set less than)
                                memory.registers[rd] = @intFromBool(@as(i64, @bitCast(memory.registers[rs1])) < @as(i64, @bitCast(memory.registers[rs2])));
                            },
                            // NOTE: This is supposed to get up to i128 which I didn't add
                            0x01 => {
                                // mulhsu (multiply high signed*unsigned)
                                // Multiplies a signed value (rs1) by an unsigned value (rs2) and returns the upper 64 bits
                                const a_i64: i64 = @bitCast(memory.registers[rs1]); // First operand is signed
                                const b_u64: u64 = memory.registers[rs2]; // Second operand is unsigned
                                const result: i128 = @as(i128, a_i64) * @as(i128, b_u64);
                                // Take the high 64 bits of the 128-bit result
                                memory.registers[rd] = @bitCast(@as(i64, @truncate(result >> 64)));
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    0x3 => {
                        switch (funct7) {
                            0x00 => {
                                // sltu (set less than unsigned)
                                memory.registers[rd] = @intFromBool(memory.registers[rs1] < memory.registers[rs2]);
                            },
                            // NOTE: This is supposed to get up to i128 which I didn't add
                            0x01 => {
                                // mulhu (multiply high unsigned)
                                // Multiplies two unsigned 64-bit values and returns the upper 64 bits
                                const a_u64: u64 = memory.registers[rs1];
                                const b_u64: u64 = memory.registers[rs2];
                                const result: u128 = @as(u128, a_u64) * @as(u128, b_u64);
                                // Take the high 64 bits of the 128-bit result
                                memory.registers[rd] = @truncate(result >> 64);
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    0x4 => {
                        switch (funct7) {
                            0x00 => {
                                // xor (bitwise XOR)
                                memory.registers[rd] = memory.registers[rs1] ^ memory.registers[rs2];
                            },
                            // NOTE: This needs fixing (actually fixed now I think)
                            0x01 => {
                                // div (divide)
                                // Implements RISC-V spec: division by zero and signed overflow
                                const dividend = @as(i64, @bitCast(memory.registers[rs1]));
                                const divisor = @as(i64, @bitCast(memory.registers[rs2]));
                                var result: i64 = undefined;
                                if (divisor == 0) {
                                    result = -1; // All bits set
                                } else if (dividend == std.math.minInt(i64) and divisor == -1) {
                                    result = dividend; // Overflow case
                                } else {
                                    result = @divTrunc(dividend, divisor);
                                }
                                memory.registers[rd] = @bitCast(result);
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    0x5 => {
                        switch (funct7) {
                            0x00 => {
                                // srl (shift right logical)
                                const shamt: u6 = @intCast(memory.registers[rs2] & 0x3f);
                                memory.registers[rd] = memory.registers[rs1] >> shamt;
                            },
                            // NOTE: This needs fixing (I think its fixed now)
                            0x01 => {
                                // divu (divide unsigned)
                                // Implements RISC-V spec: division by zero
                                const dividend = memory.registers[rs1];
                                const divisor = memory.registers[rs2];
                                var result: u64 = undefined;
                                if (divisor == 0) {
                                    result = 0xFFFF_FFFF_FFFF_FFFF;
                                } else {
                                    result = dividend / divisor;
                                }
                                memory.registers[rd] = result;
                            },
                            0x20 => {
                                // sra (shift right arithmetic)
                                const shamt: u6 = @intCast(memory.registers[rs2] & 0x3f);
                                memory.registers[rd] = @as(u64, @bitCast(@as(i64, @bitCast(memory.registers[rs1])) >> shamt));
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    0x6 => {
                        switch (funct7) {
                            0x00 => {
                                // or (bitwise OR)
                                memory.registers[rd] = memory.registers[rs1] | memory.registers[rs2];
                            },
                            // NOTE: This needs fixing (I think its fixed now)
                            0x01 => {
                                // rem (remainder)
                                const dividend = @as(i64, @bitCast(memory.registers[rs1]));
                                const divisor = @as(i64, @bitCast(memory.registers[rs2]));
                                var result: i64 = undefined;
                                if (divisor == 0) {
                                    result = dividend;
                                } else if (dividend == std.math.minInt(i64) and divisor == -1) {
                                    result = 0;
                                } else {
                                    result = @rem(dividend, divisor);
                                }
                                memory.registers[rd] = @bitCast(result);
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    0x7 => {
                        switch (funct7) {
                            0x00 => {
                                // and (bitwise AND)
                                memory.registers[rd] = memory.registers[rs1] & memory.registers[rs2];
                            },
                            // NOTE: This needs fixing (I think its fixed now)
                            0x01 => {
                                // remu (remainder unsigned)
                                const dividend = memory.registers[rs1];
                                const divisor = memory.registers[rs2];
                                var result: u64 = undefined;
                                if (divisor == 0) {
                                    result = dividend;
                                } else {
                                    result = dividend % divisor;
                                }
                                memory.registers[rd] = result;
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    else => return error.UnimplementedInstruction,
                }
            },
            0x37 => {
                // RV64I
                // lui
                // Load Upper Immediate - Loads an immediate value into the upper 20 bits of the destination register
                memory.registers[rd] = signExtend(u64, i32, inst & 0xfffff000);
            },
            0x3b => {
                // RV64M
                const funct7 = (inst >> 25) & 0x7f;

                switch (funct3) {
                    0x0 => {
                        switch (funct7) {
                            0x00 => { // addw (add word)

                                memory.registers[rd] = signExtend(u64, i32, memory.registers[rs1] +% memory.registers[rs2]);
                            },
                            0x01 => {
                                // mulw (multiply word)
                                const n1: i32 = @bitCast(@as(u32, @truncate(memory.registers[rs1])));
                                const n2: i32 = @bitCast(@as(u32, @truncate(memory.registers[rs2])));

                                memory.registers[rd] = @bitCast(@as(i64, n1 *% n2));
                            },
                            // TODO: Update this
                            0x20 => {
                                // subw (subtract word)
                                const rs1_val: u32 = @truncate(memory.registers[rs1]);
                                const rs2_val: u32 = @truncate(memory.registers[rs2]);
                                const result: i32 = @bitCast(rs1_val -% rs2_val);
                                // Sign extend to 64-bits
                                memory.registers[rd] = @bitCast(@as(i64, result));
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    0x1 => {
                        switch (funct7) {
                            // TODO: Update this
                            0x00 => {
                                // sllw (shift left logical word)
                                const shamt: u5 = @intCast(memory.registers[rs2] & 0x1f);
                                const rs1_val: u32 = @truncate(memory.registers[rs1]);
                                const shifted: u32 = rs1_val << shamt;
                                const result: i32 = @bitCast(shifted);
                                // Sign extend to 64-bits
                                memory.registers[rd] = @bitCast(@as(i64, result));
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    // TODO: Update this
                    0x4 => {
                        switch (funct7) {
                            0x01 => {
                                // divw (divide word)
                                // NOTE: This is missing overflow case via FCS
                                const rs1_val: i32 = @bitCast(@as(u32, @truncate(memory.registers[rs1])));
                                const rs2_val: i32 = @bitCast(@as(u32, @truncate(memory.registers[rs2])));

                                // Handle special cases for RISC-V:
                                // 1. Division by zero
                                // 2. Signed division overflow (-2^31 / -1)
                                var result: i32 = undefined;
                                if (rs2_val == 0) {
                                    result = -1; // Division by zero returns all 1s
                                } else if (rs1_val == -0x80000000 and rs2_val == -1) {
                                    result = -0x80000000; // Overflow case
                                } else {
                                    result = @divTrunc(rs1_val, rs2_val);
                                }

                                // Sign extend to 64-bits
                                memory.registers[rd] = @bitCast(@as(i64, result));
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    0x5 => {
                        // TODO: Update this
                        switch (funct7) {
                            0x00 => {
                                // srlw (shift right logical word)
                                const shamt: u5 = @intCast(memory.registers[rs2] & 0x1f);
                                const rs1_val: u32 = @truncate(memory.registers[rs1]);
                                const shifted: u32 = rs1_val >> shamt;
                                const result: i32 = @bitCast(shifted);
                                // Sign extend to 64-bits
                                memory.registers[rd] = @bitCast(@as(i64, result));
                            },
                            // TODO: Update this
                            0x01 => {
                                // divuw (divide unsigned word)
                                const rs1_val: u32 = @truncate(memory.registers[rs1]);
                                const rs2_val: u32 = @truncate(memory.registers[rs2]);
                                var result: u32 = undefined;
                                if (rs2_val == 0) {
                                    result = 0xFFFF_FFFF;
                                } else {
                                    result = rs1_val / rs2_val;
                                }
                                // Sign-extend the 32-bit result to 64 bits per RISC-V spec
                                memory.registers[rd] = @bitCast(@as(i64, @as(i32, @bitCast(result))));
                            },
                            // TODO: Update this
                            0x20 => {
                                // sraw (shift right arithmetic word)
                                const shamt: u5 = @intCast(memory.registers[rs2] & 0x1f);
                                const rs1_val: i32 = @bitCast(@as(u32, @truncate(memory.registers[rs1])));
                                const shifted: i32 = rs1_val >> shamt;
                                // Sign extend to 64-bits
                                memory.registers[rd] = @bitCast(@as(i64, shifted));
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    // TODO: Update this
                    0x6 => {
                        switch (funct7) {
                            0x01 => {
                                // remw (remainder word)
                                const rs1_val: i32 = @bitCast(@as(u32, @truncate(memory.registers[rs1])));
                                const rs2_val: i32 = @bitCast(@as(u32, @truncate(memory.registers[rs2])));

                                // Handle special cases for RISC-V:
                                // 1. Division by zero - remainder is the dividend
                                // 2. Signed division overflow (-2^31 / -1) - remainder is 0
                                var result: i32 = undefined;
                                if (rs2_val == 0) {
                                    result = rs1_val; // Remainder with div by zero is the dividend
                                } else if (rs1_val == -0x80000000 and rs2_val == -1) {
                                    result = 0; // Overflow case results in remainder 0
                                } else {
                                    result = @rem(rs1_val, rs2_val);
                                }

                                // Sign extend to 64-bits
                                memory.registers[rd] = @bitCast(@as(i64, result));
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    // TODO: Update this (I think its fixed now)
                    0x7 => {
                        switch (funct7) {
                            0x01 => {
                                // remuw (remainder unsigned word)
                                const rs1_val: u32 = @truncate(memory.registers[rs1]);
                                const rs2_val: u32 = @truncate(memory.registers[rs2]);
                                var result: i32 = undefined;
                                if (rs2_val == 0) {
                                    result = @bitCast(rs1_val); // sign-extend dividend
                                } else {
                                    result = @bitCast(rs1_val % rs2_val); // sign-extend result
                                }
                                // Sign extend to 64-bits
                                memory.registers[rd] = @bitCast(@as(i64, result));
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    else => return error.UnimplementedInstruction,
                }
            },
            0x43 => {
                // RV64F and RV64F
                const funct2 = (inst >> 25) & 0x3;
                const rs3 = (inst >> 27) & 0x1F;
                switch (funct2) {
                    0x0 => {
                        // fmadd.s (multiply-add single-precision)
                        try doSoftFloatOpS_Ternary(@truncate(rd), @truncate(rs1), @truncate(rs2), @truncate(rs3), false, false, memory);
                    },
                    0x1 => {
                        // fmadd.d (multiply-add double-precision)
                        try doSoftFloatOpD_Ternary(@truncate(rd), @truncate(rs1), @truncate(rs2), @truncate(rs3), false, false, memory);
                    },
                    else => return error.UnimplementedInstruction,
                }
            },
            0x47 => {
                // RV32F and RV64F
                const rs3 = (inst >> 27) & 0x1F;
                const funct2 = (inst >> 25) & 0x3;

                switch (funct2) {
                    0x0 => {
                        // fmsub.s (multiply-subtract single-precision)
                        try doSoftFloatOpS_Ternary(@truncate(rd), @truncate(rs1), @truncate(rs2), @truncate(rs3), false, true, memory);
                    },
                    0x1 => {
                        // fmsub.d (multiply-subtract double-precision)
                        try doSoftFloatOpD_Ternary(@truncate(rd), @truncate(rs1), @truncate(rs2), @truncate(rs3), false, true, memory);
                    },
                    else => return error.UnimplementedInstruction,
                }
            },
            0x4b => {
                // RV32F and RV64F
                const rs3 = (inst >> 27) & 0x1F;
                const funct2 = (inst >> 25) & 0x3;

                switch (funct2) {
                    0x0 => {
                        // fnmadd.s (negate multiply-add single-precision)
                        try doSoftFloatOpS_Ternary(@truncate(rd), @truncate(rs1), @truncate(rs2), @truncate(rs3), true, false, memory);
                    },
                    0x1 => {
                        // fnmadd.d (negate multiply-add double-precision)
                        try doSoftFloatOpD_Ternary(@truncate(rd), @truncate(rs1), @truncate(rs2), @truncate(rs3), true, false, memory);
                    },
                    else => return error.UnimplementedInstruction,
                }
            },
            0x4f => {
                // RV32F and RV64F
                const rs3 = (inst >> 27) & 0x1F;
                const funct2 = (inst >> 25) & 0x3;

                switch (funct2) {
                    0x0 => {
                        // fnmsub.s (negate multiply-subtract single-precision)
                        try doSoftFloatOpS_Ternary(@truncate(rd), @truncate(rs1), @truncate(rs2), @truncate(rs3), true, true, memory);
                    },
                    0x1 => {
                        // fnmsub.d (negate multiply-subtract double-precision)
                        try doSoftFloatOpD_Ternary(@truncate(rd), @truncate(rs1), @truncate(rs2), @truncate(rs3), true, true, memory);
                    },
                    else => return error.UnimplementedInstruction,
                }
            },
            0x53 => {
                // RV32F and RV64F
                const funct7 = (inst >> 25) & 0x7F;

                switch (funct7) {
                    0x00 => {
                        // fadd.s (add single-precision)
                        try doSoftFloatOpS(softfloat.f32_add, @truncate(rd), @truncate(rs1), @truncate(rs2), memory);
                    },
                    0x01 => {
                        // fadd.d (add double-precision)
                        try doSoftFloatOpD(softfloat.f64_add, @truncate(rd), @truncate(rs1), @truncate(rs2), memory);
                    },
                    0x04 => {
                        // fsub.s (subtract single-precision)
                        try doSoftFloatOpS(softfloat.f32_sub, @truncate(rd), @truncate(rs1), @truncate(rs2), memory);
                    },
                    0x05 => {
                        // fsub.d (subtract double-precision)
                        try doSoftFloatOpD(softfloat.f64_sub, @truncate(rd), @truncate(rs1), @truncate(rs2), memory);
                    },
                    0x08 => {
                        // fmul.s (multiply single-precision)
                        try doSoftFloatOpS(softfloat.f32_mul, @truncate(rd), @truncate(rs1), @truncate(rs2), memory);
                    },
                    0x09 => {
                        // fmul.d (multiply double-precision)
                        try doSoftFloatOpD(softfloat.f64_mul, @truncate(rd), @truncate(rs1), @truncate(rs2), memory);
                    },
                    0x0c => {
                        // fdiv.s (divide single-precision)
                        try doSoftFloatOpS(softfloat.f32_div, @truncate(rd), @truncate(rs1), @truncate(rs2), memory);
                    },
                    0x0d => {
                        // fdiv.d (divide double-precision)
                        try doSoftFloatOpD(softfloat.f64_div, @truncate(rd), @truncate(rs1), @truncate(rs2), memory);
                    },
                    0x10 => {
                        switch (funct3) {
                            0x0 => {
                                // fsgnj.s (set sign of single-precision)
                                const a_f32 = getF32FromFreg(memory.fregs[rs1]);
                                const b_f32 = getF32FromFreg(memory.fregs[rs2]);

                                // Extract the raw bits from the floating-point values
                                const a_bits = @as(u32, @bitCast(a_f32));
                                const b_bits = @as(u32, @bitCast(b_f32));

                                // Copy a's value but with b's sign bit
                                const result_bits = (a_bits & 0x7FFFFFFF) | (b_bits & 0x80000000);
                                const result_f32 = @as(f32, @bitCast(result_bits));

                                // NaN-box the result for 64-bit register
                                memory.fregs[rd] = @as(f64, @bitCast((@as(u64, @as(u32, @bitCast(result_f32))) | 0xFFFFFFFF00000000)));
                            },
                            0x1 => {
                                // fsgnjn.s (set inverted sign of single-precision)
                                const a_f32 = getF32FromFreg(memory.fregs[rs1]);
                                const b_f32 = getF32FromFreg(memory.fregs[rs2]);

                                // Extract the raw bits from the floating-point values
                                const a_bits = @as(u32, @bitCast(a_f32));
                                const b_bits = @as(u32, @bitCast(b_f32));

                                // Copy a's value but with inverted b's sign bit
                                const result_bits = (a_bits & 0x7FFFFFFF) | (~b_bits & 0x80000000);
                                const result_f32 = @as(f32, @bitCast(result_bits));

                                // NaN-box the result for 64-bit register
                                memory.fregs[rd] = @as(f64, @bitCast((@as(u64, @as(u32, @bitCast(result_f32))) | 0xFFFFFFFF00000000)));
                            },
                            0x2 => {
                                // fsgnjx.s (set sign of single-precision using XOR)
                                const a_f32 = getF32FromFreg(memory.fregs[rs1]);
                                const b_f32 = getF32FromFreg(memory.fregs[rs2]);

                                // Extract the raw bits from the floating-point values
                                const a_bits = @as(u32, @bitCast(a_f32));
                                const b_bits = @as(u32, @bitCast(b_f32));

                                // Copy a's value but XOR the sign bits
                                const result_bits = (a_bits & 0x7FFFFFFF) | ((a_bits ^ b_bits) & 0x80000000);
                                const result_f32 = @as(f32, @bitCast(result_bits));

                                // NaN-box the result for 64-bit register
                                memory.fregs[rd] = @as(f64, @bitCast((@as(u64, @as(u32, @bitCast(result_f32))) | 0xFFFFFFFF00000000)));
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    0x11 => {
                        switch (funct3) {
                            0x0 => {
                                // fsgnj.d (set sign of double-precision)
                                const a_f64 = memory.fregs[rs1];
                                const b_f64 = memory.fregs[rs2];

                                // Extract the raw bits from the floating-point values
                                const a_bits = @as(u64, @bitCast(a_f64));
                                const b_bits = @as(u64, @bitCast(b_f64));

                                // Copy a's value but with b's sign bit
                                const result_bits = (a_bits & 0x7FFFFFFFFFFFFFFF) | (b_bits & 0x8000000000000000);
                                memory.fregs[rd] = @as(f64, @bitCast(result_bits));
                            },
                            0x1 => {
                                // fsgnjn.d (set inverted sign of double-precision)
                                const a_f64 = memory.fregs[rs1];
                                const b_f64 = memory.fregs[rs2];

                                // Extract the raw bits from the floating-point values
                                const a_bits = @as(u64, @bitCast(a_f64));
                                const b_bits = @as(u64, @bitCast(b_f64));

                                // Copy a's value but with inverted b's sign bit
                                const result_bits = (a_bits & 0x7FFFFFFFFFFFFFFF) | (~b_bits & 0x8000000000000000);
                                memory.fregs[rd] = @as(f64, @bitCast(result_bits));
                            },
                            0x2 => {
                                // fsgnjx.d (set sign of double-precision using XOR)
                                const a_f64 = memory.fregs[rs1];
                                const b_f64 = memory.fregs[rs2];

                                // Extract the raw bits from the floating-point values
                                const a_bits = @as(u64, @bitCast(a_f64));
                                const b_bits = @as(u64, @bitCast(b_f64));

                                // Copy a's value but XOR the sign bits
                                const result_bits = (a_bits & 0x7FFFFFFFFFFFFFFF) | ((a_bits ^ b_bits) & 0x8000000000000000);
                                memory.fregs[rd] = @as(f64, @bitCast(result_bits));
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    0x14 => {
                        switch (funct3) {
                            0x0 => {
                                // fmin.s (minimum single-precision)
                                try doSoftFloatOpS(softfloat.f32_min, @truncate(rd), @truncate(rs1), @truncate(rs2), memory);
                            },
                            0x1 => {
                                // fmax.s (maximum single-precision)
                                try doSoftFloatOpS(softfloat.f32_max, @truncate(rd), @truncate(rs1), @truncate(rs2), memory);
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    0x15 => {
                        switch (funct3) {
                            0x0 => {
                                // fmin.d (minimum double-precision)
                                try doSoftFloatOpD(softfloat.f64_min, @truncate(rd), @truncate(rs1), @truncate(rs2), memory);
                            },
                            0x1 => {
                                // fmax.d (maximum double-precision)
                                try doSoftFloatOpD(softfloat.f64_max, @truncate(rd), @truncate(rs1), @truncate(rs2), memory);
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    0x20 => {
                        // fcvt.s.d (convert double-precision to single-precision)
                        try doSoftFloatConvertDtoS(@truncate(rd), @truncate(rs1), memory);
                    },
                    0x21 => {
                        // fcvt.d.s (convert single-precision to double-precision)
                        try doSoftFloatConvertStoD(@truncate(rd), @truncate(rs1), memory);
                    },
                    0x2c => {
                        // fsqrt.s (square root single-precision)
                        try doSoftFloatOpS_Unary(softfloat.f32_sqrt, @truncate(rd), @truncate(rs1), memory);
                    },
                    0x2d => {
                        // fsqrt.d (square root double-precision)
                        try doSoftFloatOpD_Unary(softfloat.f64_sqrt, @truncate(rd), @truncate(rs1), memory);
                    },
                    0x50 => {
                        switch (funct3) {
                            0x0 => {
                                // fle.s (less than or equal to single-precision)
                                try doSoftFloatCompareS(softfloat.f32_le, @truncate(rd), @truncate(rs1), @truncate(rs2), memory);
                            },
                            0x1 => {
                                // flt.s (less than single-precision)
                                try doSoftFloatCompareS(softfloat.f32_lt, @truncate(rd), @truncate(rs1), @truncate(rs2), memory);
                            },
                            0x2 => {
                                // feq.s (equal single-precision)
                                try doSoftFloatCompareS(softfloat.f32_eq, @truncate(rd), @truncate(rs1), @truncate(rs2), memory);
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    0x51 => {
                        switch (funct3) {
                            0x0 => {
                                // fle.d (less than or equal to double-precision)
                                try doSoftFloatCompareD(softfloat.f64_le, @truncate(rd), @truncate(rs1), @truncate(rs2), memory);
                            },
                            0x1 => {
                                // flt.d (less than double-precision)
                                try doSoftFloatCompareD(softfloat.f64_lt, @truncate(rd), @truncate(rs1), @truncate(rs2), memory);
                            },
                            0x2 => {
                                // feq.d (equal double-precision)
                                try doSoftFloatCompareD(softfloat.f64_eq, @truncate(rd), @truncate(rs1), @truncate(rs2), memory);
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    0x60 => { // fcvt from S
                        const dest_type_code = rs2; // Encodes destination integer type W/WU/L/LU
                        switch (dest_type_code) {
                            0x0 => { // fcvt.w.s (convert single-precision to signed word)
                                try doSoftFloatConvertStoI(i32, ReturnTypeOf(softfloat.f32_to_i32), softfloat.f32_to_i32, @truncate(rd), @truncate(rs1), memory);
                            },
                            0x1 => { // fcvt.wu.s (convert single-precision to unsigned word)
                                try doSoftFloatConvertStoI(u32, ReturnTypeOf(softfloat.f32_to_ui32), softfloat.f32_to_ui32, @truncate(rd), @truncate(rs1), memory);
                            },
                            0x2 => { // fcvt.l.s (convert single-precision to signed long)
                                try doSoftFloatConvertStoI(i64, i64, softfloat.f32_to_i64, @truncate(rd), @truncate(rs1), memory);
                            },
                            0x3 => { // fcvt.lu.s (convert single-precision to unsigned long)
                                try doSoftFloatConvertStoI(u64, u64, softfloat.f32_to_ui64, @truncate(rd), @truncate(rs1), memory);
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    0x61 => { // fcvt from D
                        const dest_type_code = rs2;
                        switch (dest_type_code) {
                            0x0 => { // fcvt.w.d (convert double-precision to signed word)
                                try doSoftFloatConvertDtoI(i32, ReturnTypeOf(softfloat.f64_to_i32), softfloat.f64_to_i32, @truncate(rd), @truncate(rs1), memory);
                            },
                            0x1 => { // fcvt.wu.d (convert double-precision to unsigned word)
                                try doSoftFloatConvertDtoI(u32, ReturnTypeOf(softfloat.f64_to_ui32), softfloat.f64_to_ui32, @truncate(rd), @truncate(rs1), memory);
                            },
                            0x2 => { // fcvt.l.d (convert double-precision to signed long)
                                try doSoftFloatConvertDtoI(i64, i64, softfloat.f64_to_i64, @truncate(rd), @truncate(rs1), memory);
                            },
                            0x3 => { // fcvt.lu.d (convert double-precision to unsigned long)
                                try doSoftFloatConvertDtoI(u64, u64, softfloat.f64_to_ui64, @truncate(rd), @truncate(rs1), memory);
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    0x68 => { // fcvt to S
                        const source_type_code = rs2;
                        switch (source_type_code) {
                            0x0 => { // fcvt.s.w (convert signed word to single-precision)
                                try doSoftFloatConvertItoS(i32, softfloat.i32_to_f32, @truncate(rd), @truncate(rs1), memory);
                            },
                            0x1 => { // fcvt.s.wu (convert unsigned word to single-precision)
                                try doSoftFloatConvertItoS(u32, softfloat.ui32_to_f32, @truncate(rd), @truncate(rs1), memory);
                            },
                            0x2 => { // fcvt.s.l (convert signed long to single-precision)
                                try doSoftFloatConvertItoS(i64, softfloat.i64_to_f32, @truncate(rd), @truncate(rs1), memory);
                            },
                            0x3 => { // fcvt.s.lu (convert unsigned long to single-precision)
                                try doSoftFloatConvertItoS(u64, softfloat.ui64_to_f32, @truncate(rd), @truncate(rs1), memory);
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    0x69 => { // fcvt to D
                        const source_type_code = rs2;
                        switch (source_type_code) {
                            0x0 => { // fcvt.d.w (convert signed word to double-precision)
                                try doSoftFloatConvertItoD(i32, softfloat.i32_to_f64, @truncate(rd), @truncate(rs1), memory);
                            },
                            0x1 => { // fcvt.d.wu (convert unsigned word to double-precision)
                                try doSoftFloatConvertItoD(u32, softfloat.ui32_to_f64, @truncate(rd), @truncate(rs1), memory);
                            },
                            0x2 => { // fcvt.d.l (convert signed long to double-precision)
                                try doSoftFloatConvertItoD(i64, softfloat.i64_to_f64, @truncate(rd), @truncate(rs1), memory);
                            },
                            0x3 => { // fcvt.d.lu (convert unsigned long to double-precision)
                                try doSoftFloatConvertItoD(u64, softfloat.ui64_to_f64, @truncate(rd), @truncate(rs1), memory);
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    0x70 => {
                        switch (funct3) {
                            0x0 => {
                                // fmv.x.w (move word from float to int, sign-extended)
                                // Get the raw f32 value properly considering NaN-boxing
                                const f32_val = getF32FromFreg(memory.fregs[rs1]);
                                // Direct bit transfer without conversion
                                const raw_bits = @as(u32, @bitCast(f32_val));
                                // Sign-extend to 64 bits
                                memory.registers[rd] = signExtend(u64, i32, raw_bits);
                            },
                            // TODO: Make sure this is good
                            0x1 => {
                                // fclass.s (classify single-precision)

                                const f32_val = getF32FromFreg(memory.fregs[rs1]);
                                const sf32 = @as(softfloat.float32_t, @bitCast(f32_val));
                                memory.registers[rd] = softfloat.f32_classify(sf32);
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    0x71 => {
                        switch (funct3) {
                            // TODO: Fix this
                            0x0 => {
                                // fmv.x.d (move double-precision to integer)
                                const f64_bits = @as(u64, @bitCast(memory.fregs[rs1]));
                                // Direct bit transfer without conversion
                                memory.registers[rd] = f64_bits;
                            },
                            // TODO: Make sure this is good
                            0x1 => {
                                // fclass.d (classify double-precision)
                                const sf64 = @as(softfloat.float64_t, @bitCast(memory.fregs[rs1]));
                                memory.registers[rd] = softfloat.f64_classify(sf64);
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    0x78 => {
                        // fmv.w.x (move word from integer to float, NaN-boxing)
                        // Extract the lower 32 bits from the integer register
                        const word_bits = @as(u32, @truncate(memory.registers[rs1]));
                        // Create a NaN-boxed representation (upper 32 bits all 1s, lower 32 bits from register)
                        const nan_boxed_bits = (@as(u64, 0xFFFFFFFF) << 32) | @as(u64, word_bits);
                        // Store as a f64 value
                        memory.fregs[rd] = @as(f64, @bitCast(nan_boxed_bits));
                    },
                    // NOTE: Not 100% sure about this
                    0x79 => {
                        // fmv.d.x (move integer to double-precision)
                        const int_bits = memory.registers[rs1];
                        // Direct bit transfer without conversion
                        memory.fregs[rd] = @as(f64, @bitCast(int_bits));
                    },
                    else => {
                        return error.UnimplementedInstruction;
                    },
                }
            },
            0x63 => { // B-type (branches)
                // imm[12|10:5|4:1|11] = inst[31|30:25|11:8|7]

                const imm = (@as(u64, @bitCast((@as(i64, @as(i32, @bitCast((inst & 0x80000000)))) >> 19)))) | ((inst & 0x80) << 4) // imm[11]
                | ((inst >> 20) & 0x7e0) // imm[10:5]
                | ((inst >> 7) & 0x1e); // imm[4:1]

                switch (funct3) {
                    0x0 => {
                        // beq
                        if (memory.registers[rs1] == memory.registers[rs2]) {
                            memory.pc = memory.pc +% imm;
                            return true;
                        }
                    },
                    0x1 => {
                        // bne
                        if (memory.registers[rs1] != memory.registers[rs2]) {
                            memory.pc = memory.pc +% imm;
                            return true;
                        }
                    },
                    0x4 => {
                        // blt
                        if (@as(i64, @bitCast(memory.registers[rs1])) < (@as(i64, @bitCast(memory.registers[rs2])))) {
                            memory.pc = memory.pc +% imm;
                            return true;
                        }
                    },
                    0x5 => {
                        // bge
                        if (@as(i64, @bitCast(memory.registers[rs1])) >= (@as(i64, @bitCast(memory.registers[rs2])))) {
                            memory.pc = memory.pc +% imm;
                            return true;
                        }
                    },
                    0x6 => {
                        // bltu
                        if (memory.registers[rs1] < memory.registers[rs2]) {
                            memory.pc = memory.pc +% imm;
                            return true;
                        }
                    },
                    0x7 => {
                        // bgeu
                        if (memory.registers[rs1] >= memory.registers[rs2]) {
                            memory.pc = memory.pc +% imm;
                            return true;
                        }
                    },
                    else => return error.UnimplementedInstruction,
                }
            },
            0x67 => { // JALR (I-type)
                // // Jump and Link Register
                // Jump - Jumps to a target address (calculated from the immediate relative to the register passed in)
                // Link - Stores the return address in a register (For returning from a function)
                const t = memory.pc +% 4;

                const offset = @as(i64, @as(i32, @bitCast(inst))) >> 20;
                const target = (@as(i64, @bitCast(memory.registers[rs1])) +% offset) & ~@as(i64, 1);

                memory.pc = @as(u64, @bitCast(target));

                memory.registers[rd] = t;
                return true;
            },
            0x6F => { // JAL (J-type)
                // Jump and Link
                // Jump - Jumps to a target address (calculated from the immediate relative to the pc)
                // Link - Stores the return address in a register (For returning from a function)
                memory.registers[rd] = memory.pc +% 4;

                // imm[20|10:1|11|19:12] = inst[31|30:21|20|19:12]
                const offset = @as(u64, @bitCast((@as(i64, @as(i32, @bitCast((inst & 0x80000000)))) >> 11))) // imm[20]
                    | (inst & 0xff000) // imm[19:12]
                    | ((inst >> 9) & 0x800) // imm[11]
                    | ((inst >> 20) & 0x7fe); // imm[10:1]

                memory.pc = memory.pc +% offset;
                return true;
            },
            0x73 => {
                const csr_addr = @as(u16, @truncate((inst >> 20) & 0xfff));
                const funct7 = (inst >> 25) & 0x7F;

                switch (funct3) {
                    0x0 => {
                        switch (funct7) {
                            0x0 => {
                                switch (rs2) {
                                    0x0 => {
                                        // ecall
                                        try self.handleEcall(memory, user_data);
                                    },
                                    0x1 => {
                                        // ebreak
                                        // Return breakpoint error to be handled by the execution loop
                                        return error.Breakpoint;
                                    },
                                    0x2 => {
                                        // uret
                                        // Not implemented
                                    },
                                    else => return error.UnimplementedInstruction,
                                }
                            },
                            0x8 => {
                                switch (rs2) {
                                    0x2 => {
                                        // sret
                                        // "The RISC-V Reader" book says:
                                        // "Returns from a supervisor-mode exception handler. Sets the pc to
                                        // CSRs[sepc], the privilege mode to CSRs[sstatus].SPP,
                                        // CSRs[sstatus].SIE to CSRs[sstatus].SPIE, CSRs[sstatus].SPIE to
                                        // 1, and CSRs[sstatus].SPP to 0."

                                        // Invalidate LR/SC reservations on privilege mode change
                                        self.invalidateReservation();

                                        // Set the program counter to the supervisor exception program counter (SEPC)
                                        memory.pc = memory.csr.read(Csr.SEPC) -% 4;

                                        // Set the current privileged mode depending on a previous
                                        // privilege mode for supervisor mode (SPP, 8)
                                        const spp = memory.csr.readSstatus(Csr.XSTATUS_SPP);
                                        switch (spp) {
                                            0 => {
                                                // TODO: When mode enum is available, set to User
                                                memory.privilege_mode = Mode.User;
                                            },
                                            1 => {
                                                // If SPP != M-mode, SRET also sets MPRV=0
                                                memory.csr.writeMstatus(Csr.MSTATUS_MPRV, 0);
                                                // TODO: When mode enum is available, set to Supervisor
                                                memory.privilege_mode = Mode.Supervisor;
                                            },
                                            else => {
                                                // Should not happen, but handle anyway
                                                // TODO: When mode enum is available, set to Debug
                                                memory.privilege_mode = Mode.Debug;
                                            },
                                        }

                                        // Read SPIE and set SIE to it
                                        memory.csr.writeSstatus(Csr.XSTATUS_SIE, memory.csr.readSstatus(Csr.XSTATUS_SPIE));

                                        // Set SPIE to 1
                                        memory.csr.writeSstatus(Csr.XSTATUS_SPIE, 1);

                                        // Set SPP to 0 (User mode)
                                        memory.csr.writeSstatus(Csr.XSTATUS_SPP, 0);
                                    },
                                    0x5 => {
                                        // wfi
                                        // Not implemented
                                    },
                                    else => return error.UnimplementedInstruction,
                                }
                            },
                            0x9 => {
                                // sfence.vma
                                // Not implemented
                            },
                            0x11 => {
                                // hfence.bvma
                                // Not implemented
                            },
                            0x18 => {
                                switch (rs2) {
                                    0x2 => {
                                        // mret
                                        // "The RISC-V Reader" book says:
                                        // "Returns from a machine-mode exception handler. Sets the pc to
                                        // CSRs[mepc], the privilege mode to CSRs[mstatus].MPP,
                                        // CSRs[mstatus].MIE to CSRs[mstatus].MPIE, and CSRs[mstatus].MPIE
                                        // to 1; and, if user mode is supported, sets CSRs[mstatus].MPP to
                                        // 0".

                                        // Invalidate LR/SC reservations on privilege mode change
                                        self.invalidateReservation();

                                        // Set the program counter to the machine exception program counter (MEPC)
                                        memory.pc = memory.csr.read(Csr.MEPC) -% 4;

                                        // Set the current privileged mode depending on a previous
                                        // privilege mode for machine mode (MPP, 11..12)
                                        const mpp = memory.csr.readMstatus(Csr.MSTATUS_MPP);
                                        switch (mpp) {
                                            0 => { // User mode
                                                // If MPP != M-mode, MRET also sets MPRV=0
                                                memory.csr.writeMstatus(Csr.MSTATUS_MPRV, 0);
                                                // TODO: When mode enum is available, set to User
                                                memory.privilege_mode = Mode.User;
                                            },
                                            1 => { // Supervisor mode
                                                // If MPP != M-mode, MRET also sets MPRV=0
                                                memory.csr.writeMstatus(Csr.MSTATUS_MPRV, 0);
                                                // TODO: When mode enum is available, set to Supervisor
                                                memory.privilege_mode = Mode.Supervisor;
                                            },
                                            3 => { // Machine mode
                                                // TODO: When mode enum is available, set to Machine
                                                memory.privilege_mode = Mode.Machine;
                                            },
                                            else => {
                                                // Should not happen, but handle anyway
                                                // TODO: When mode enum is available, set to Debug
                                                memory.privilege_mode = Mode.Debug;
                                            },
                                        }

                                        // Read MPIE and set MIE to it
                                        memory.csr.writeMstatus(Csr.MSTATUS_MIE, memory.csr.readMstatus(Csr.MSTATUS_MPIE));

                                        // Set MPIE to 1
                                        memory.csr.writeMstatus(Csr.MSTATUS_MPIE, 1);

                                        // Set MPP to 0 (User mode)
                                        memory.csr.writeMstatus(Csr.MSTATUS_MPP, 0);
                                    },
                                    else => return error.UnimplementedInstruction,
                                }
                            },
                            0x51 => {
                                // hfence.gvma
                                // Not implemented
                            },
                            else => return error.UnimplementedInstruction,
                        }
                    },
                    // No clue if csrr support works
                    0x1 => {
                        // csrrw - Atomic Read/Write CSR
                        const old_value = memory.csr.read(csr_addr);
                        if (rd != 0) {
                            memory.registers[rd] = old_value;
                        }
                        if (rs1 != 0) {
                            memory.csr.write(csr_addr, memory.registers[rs1]);
                        }

                        if (csr_addr == Csr.SATP) {
                            // TODO: Implement update_paging when needed
                            // self.update_paging();
                        }
                    },
                    0x2 => {
                        // csrrs - Atomic Read and Set Bits in CSR
                        const old_value = memory.csr.read(csr_addr);
                        memory.registers[rd] = old_value;

                        // Only write if rs1 != 0 (otherwise it's just a read)
                        if (rs1 != 0) {
                            const new_value = old_value | memory.registers[rs1];
                            memory.csr.write(csr_addr, new_value);

                            if (csr_addr == Csr.SATP) {
                                // TODO: Implement update_paging when needed
                                // self.update_paging();
                            }
                        }
                    },
                    0x3 => {
                        // csrrc - Atomic Read and Clear Bits in CSR
                        const old_value = memory.csr.read(csr_addr);
                        memory.registers[rd] = old_value;

                        // Only write if rs1 != 0 (otherwise it's just a read)
                        if (rs1 != 0) {
                            const new_value = old_value & ~memory.registers[rs1];
                            memory.csr.write(csr_addr, new_value);

                            if (csr_addr == Csr.SATP) {
                                // TODO: Implement update_paging when needed
                                // self.update_paging();
                            }
                        }
                    },
                    0x5 => {
                        // csrrwi - Atomic Read/Write CSR (Immediate)
                        const zimm = rs1; // rs1 field interpreted as a 5-bit zero-extended immediate
                        const old_value = memory.csr.read(csr_addr);

                        if (rd != 0) {
                            memory.registers[rd] = old_value;
                        }

                        memory.csr.write(csr_addr, zimm);

                        if (csr_addr == Csr.SATP) {
                            // TODO: Implement update_paging when needed
                            // self.update_paging();
                        }
                    },
                    0x6 => {
                        // csrrsi - Atomic Read and Set Bits in CSR (Immediate)
                        const zimm = rs1; // rs1 field interpreted as a 5-bit zero-extended immediate
                        const old_value = memory.csr.read(csr_addr);

                        memory.registers[rd] = old_value;

                        // Only write if zimm != 0 (otherwise it's just a read)
                        if (zimm != 0) {
                            const new_value = old_value | zimm;
                            memory.csr.write(csr_addr, new_value);

                            if (csr_addr == Csr.SATP) {
                                // TODO: Implement update_paging when needed
                                // self.update_paging();
                            }
                        }
                    },
                    0x7 => {
                        // csrrci - Atomic Read and Clear Bits in CSR (Immediate)
                        const zimm = rs1; // rs1 field interpreted as a 5-bit zero-extended immediate
                        const old_value = memory.csr.read(csr_addr);

                        memory.registers[rd] = old_value;

                        // Only write if zimm != 0 (otherwise it's just a read)
                        if (zimm != 0) {
                            const new_value = old_value & ~zimm;
                            memory.csr.write(csr_addr, new_value);

                            if (csr_addr == Csr.SATP) {
                                // TODO: Implement update_paging when needed
                                // self.update_paging();
                            }
                        }
                    },
                    else => return error.UnimplementedInstruction,
                }
            },
            else => {
                print("0x{x:0>8}\n", .{inst});
                // Binary representation of the instruction
                // std.debug.print("Unimplemented instruction: opcode={an}, rd={x}, rs1={x}, rs2={x}, funct3={x:01}\n", .{ opcode, rd, rs1, rs2, funct3 });
                return error.UnimplementedInstruction;
            },
        }
        return false;
    }

    fn handleEcall(self: *RiscVCpu, address_space: *AddressSpace, user_data: ?*anyopaque) !void {
        // Invalidate any LR/SC reservations on system calls
        self.invalidateReservation();

        // RISC-V syscall number is in a7 (x17)
        const syscall_num = address_space.registers[17];

        // Arguments are in a0-a5 (x10-x15)
        const args = [6]u64{
            address_space.registers[10], // a0
            address_space.registers[11], // a1
            address_space.registers[12], // a2
            address_space.registers[13], // a3
            address_space.registers[14], // a4
            address_space.registers[15], // a5
        };

        // If we have a hook registered, call it
        if (address_space.ecall_hook) |hook| {
            // Call the hook function and set a0 (x10) to the return value
            const result = hook(address_space, syscall_num, args, user_data);

            // Sentinel returned by hooks that replaced the process image
            // (execve); the PC already points at the new entry point.
            const EIMAGE_REPLACED = @as(u64, @bitCast(@as(i64, -315)));

            if (result == EIMAGE_REPLACED) {
                // This means we replaced the process image, so we need to set the PC to the new process image
                return;
            }

            const EBLOCKED = @as(u64, @bitCast(@as(i64, -242)));

            // We implement blocking by returning -242, so we need to go back to the instruction that called ecall
            if (result == EBLOCKED) {
                // address_space.pc -= 4; // Remove?
                // Rewind PC so that once the thread is rescheduled the ECALL
                // instruction will be re-executed just like Linux does when
                // it returns -EINTR / -EAGAIN and user space restarts the
                // syscall.
                return;
            }

            address_space.registers[10] = result;
        } else {
            // Default ECALL implementation (can be expanded later)
            // Currently does nothing
            print("ECALL: syscall_num={d}, args=[{d}, {d}, {d}, {d}, {d}, {d}]\n", .{ syscall_num, args[0], args[1], args[2], args[3], args[4], args[5] });

            @panic("ECALL hook not found");
        }
    }

    // Helper function to update floating-point flags after operations
    // pub fn updateFloatFlags(self: *RiscVCpu, orig_a: f32, orig_b: f32, result: f32) void {
    //     // Check for inexact result (most common flag)
    //     // IEEE 754: If the rounded result is not exact or if it overflows without an overflow trap
    //     const exact_result = @as(f64, @floatCast(orig_a)) + @as(f64, @floatCast(orig_b));
    //     const rounded_back = @as(f32, @floatCast(exact_result));

    //     if (result != rounded_back) {
    //         // Set inexact flag
    //         self.csr.csrs[Csr.FFLAGS] |= Csr.NX;
    //     }

    //     // Check for overflow
    //     if (result == std.math.inf(f32) or result == -std.math.inf(f32)) {
    //         // Set overflow flag
    //         self.csr.csrs[Csr.FFLAGS] |= Csr.OF;
    //         // Overflow also implies inexact
    //         self.csr.csrs[Csr.FFLAGS] |= Csr.NX;
    //     }

    //     // Check for underflow
    //     const min_normal = @as(f32, @bitCast(@as(u32, 0x00800000))); // Min normal positive value
    //     if ((result != 0.0) and (@abs(result) < min_normal)) {
    //         // Set underflow flag
    //         self.csr.csrs[Csr.FFLAGS] |= Csr.UF;
    //         // Underflow typically implies inexact
    //         self.csr.csrs[Csr.FFLAGS] |= Csr.NX;
    //     }

    //     // Update FCSR (combines FRM and FFLAGS)
    //     self.csr.csrs[Csr.FCSR] = (self.csr.csrs[Csr.FRM] << 5) | self.csr.csrs[Csr.FFLAGS];
    // }

    // pub fn updateFloatFlagsDouble(self: *RiscVCpu, orig_a: f64, orig_b: f64, result: f64) void {
    //     // Check for inexact result (most common flag)
    //     // This is simplified - in a real implementation you'd want more precise checks
    //     // const exact_result = orig_a + orig_b;
    //     _ = orig_a;
    //     _ = orig_b;

    //     // An inexact result often occurs in floating point operations
    //     // For simplicity, we're setting it for most operations
    //     self.csr.csrs[Csr.FFLAGS] |= Csr.NX;

    //     // Check for overflow
    //     if (result == std.math.inf(f64) or result == -std.math.inf(f64)) {
    //         // Set overflow flag
    //         self.csr.csrs[Csr.FFLAGS] |= Csr.OF;
    //     }

    //     // Check for underflow
    //     const min_normal = @as(f64, @bitCast(@as(u64, 0x0010000000000000))); // Min normal positive value
    //     if ((result != 0.0) and (@abs(result) < min_normal)) {
    //         // Set underflow flag
    //         self.csr.csrs[Csr.FFLAGS] |= Csr.UF;
    //     }

    //     // Update FCSR (combines FRM and FFLAGS)
    //     self.csr.csrs[Csr.FCSR] = (self.csr.csrs[Csr.FRM] << 5) | self.csr.csrs[Csr.FFLAGS];
    // }
};
fn getF32FromFreg(val: f64) f32 {
    const bits = @as(u64, @bitCast(val));
    if ((bits >> 32) == 0xFFFFFFFF) {
        // NaN-boxed: extract lower 32 bits as f32
        return @as(f32, @bitCast(@as(u32, @intCast(bits & 0xFFFFFFFF))));
    } else {
        // Not NaN-boxed: return canonical NaN for f32
        return @as(f32, @bitCast(@as(u32, @intCast(0x7fc00000))));
    }
}

// Helper function to map RISC-V rounding mode to Softfloat rounding mode
fn getSoftfloatRoundingMode(rm: u64) !u8 {
    return switch (rm) {
        0 => softfloat.softfloat_round_near_even, // RNE
        1 => softfloat.softfloat_round_minMag, // RTZ
        2 => softfloat.softfloat_round_min, // RDN
        3 => softfloat.softfloat_round_max, // RUP
        4 => softfloat.softfloat_round_near_maxMag, // RMM
        7 => @panic("Dynamic rounding mode not yet supported"), // DYN
        else => return error.IllegalInstruction, // Invalid rounding mode
    };
}

// Helper function to update FFLAGS based on softfloat exceptions
fn updateFFlagsFromSoftfloat(memory: *AddressSpace) void {
    const flags = softfloat.softfloat_exceptionFlags & 0b11111; // 0b1111 masks to 5 bits

    // Directly map softfloat flags to RISC-V FFLAGS
    memory.csr.csrs[Csr.FFLAGS] = @as(u64, flags); // Write only allowable bits

    // Update FCSR to reflect new FFLAGS and current FRM
    const frm = (memory.csr.csrs[Csr.FRM] & 0b111) << 5;
    const fflags = memory.csr.csrs[Csr.FFLAGS] & 0b11111;
    memory.csr.csrs[Csr.FCSR] = frm | fflags;

    softfloat.softfloat_exceptionFlags = 0;
}

// Helper for single-precision operations using softfloat
fn doSoftFloatOpS(
    comptime op_func: fn (softfloat.float32_t, softfloat.float32_t) callconv(.c) softfloat.float32_t,
    rd: u5,
    rs1: u5,
    rs2: u5,
    memory: *AddressSpace,
) !void {
    const rm = try getSoftfloatRoundingMode(memory.csr.read(Csr.FRM));
    softfloat.softfloat_roundingMode = rm;

    const a_f32 = getF32FromFreg(memory.fregs[rs1]);
    const b_f32 = getF32FromFreg(memory.fregs[rs2]);

    const a_sf32 = @as(softfloat.float32_t, @bitCast(a_f32));
    const b_sf32 = @as(softfloat.float32_t, @bitCast(b_f32));

    const result_sf32 = op_func(a_sf32, b_sf32); // Call op without rm
    updateFFlagsFromSoftfloat(memory);

    const result_f32: f32 = @bitCast(result_sf32);
    // NaN-box the result
    memory.fregs[rd] = @as(f64, @bitCast((@as(u64, @as(u32, @bitCast(result_f32))) | 0xFFFFFFFF00000000)));
}

// Helper for unary single-precision operations using softfloat
fn doSoftFloatOpS_Unary(
    comptime op_func: fn (softfloat.float32_t) callconv(.c) softfloat.float32_t,
    rd: u5,
    rs1: u5,
    memory: *AddressSpace,
) !void {
    const rm = try getSoftfloatRoundingMode(memory.csr.read(Csr.FRM));
    softfloat.softfloat_roundingMode = rm;

    const a_f32 = getF32FromFreg(memory.fregs[rs1]);
    const a_sf32 = @as(softfloat.float32_t, @bitCast(a_f32));

    const result_sf32 = op_func(a_sf32); // Call unary op
    updateFFlagsFromSoftfloat(memory);

    const result_f32: f32 = @bitCast(result_sf32);
    // NaN-box the result
    memory.fregs[rd] = @as(f64, @bitCast((@as(u64, @as(u32, @bitCast(result_f32))) | 0xFFFFFFFF00000000)));
}

// Helper for double-precision operations using softfloat
fn doSoftFloatOpD(
    comptime op_func: fn (softfloat.float64_t, softfloat.float64_t) callconv(.c) softfloat.float64_t,
    rd: u5,
    rs1: u5,
    rs2: u5,
    memory: *AddressSpace,
) !void {
    const rm = try getSoftfloatRoundingMode(memory.csr.read(Csr.FRM));
    softfloat.softfloat_roundingMode = rm;

    const a_f64 = memory.fregs[rs1];
    const b_f64 = memory.fregs[rs2];

    const a_sf64 = @as(softfloat.float64_t, @bitCast(a_f64));
    const b_sf64 = @as(softfloat.float64_t, @bitCast(b_f64));

    const result_sf64 = op_func(a_sf64, b_sf64); // Call op without rm
    updateFFlagsFromSoftfloat(memory);

    memory.fregs[rd] = @as(f64, @bitCast(result_sf64));
}

// Helper for unary double-precision operations using softfloat
fn doSoftFloatOpD_Unary(
    comptime op_func: fn (softfloat.float64_t) callconv(.c) softfloat.float64_t,
    rd: u5,
    rs1: u5,
    memory: *AddressSpace,
) !void {
    const rm = try getSoftfloatRoundingMode(memory.csr.read(Csr.FRM));
    softfloat.softfloat_roundingMode = rm;

    const a_f64 = memory.fregs[rs1];
    const a_sf64 = @as(softfloat.float64_t, @bitCast(a_f64));

    const result_sf64 = op_func(a_sf64);
    updateFFlagsFromSoftfloat(memory);

    memory.fregs[rd] = @as(f64, @bitCast(result_sf64));
}

// Helper for single-precision comparisons using softfloat
fn doSoftFloatCompareS(
    comptime cmp_func: fn (softfloat.float32_t, softfloat.float32_t) callconv(.c) bool, // Change i32 to bool
    rd: u5,
    rs1: u5,
    rs2: u5,
    memory: *AddressSpace,
) !void {
    const rm = try getSoftfloatRoundingMode(memory.csr.read(Csr.FRM));
    softfloat.softfloat_roundingMode = rm;

    const a_f32 = getF32FromFreg(memory.fregs[rs1]);
    const b_f32 = getF32FromFreg(memory.fregs[rs2]);

    const a_sf32 = @as(softfloat.float32_t, @bitCast(a_f32));
    const b_sf32 = @as(softfloat.float32_t, @bitCast(b_f32));

    // Pass 'false' for signaling NaN handling, as per RISC-V spec for fclass/fcmp
    const result = cmp_func(a_sf32, b_sf32);
    updateFFlagsFromSoftfloat(memory);

    memory.registers[rd] = @intFromBool(result); // Use @intFromBool
}

// Helper for double-precision comparisons using softfloat
fn doSoftFloatCompareD(
    comptime cmp_func: fn (softfloat.float64_t, softfloat.float64_t) callconv(.c) bool, // Change i32 to bool
    rd: u5,
    rs1: u5,
    rs2: u5,
    memory: *AddressSpace,
) !void {
    const rm = try getSoftfloatRoundingMode(memory.csr.read(Csr.FRM));
    softfloat.softfloat_roundingMode = rm;

    const a_f64 = memory.fregs[rs1];
    const b_f64 = memory.fregs[rs2];

    const a_sf64 = @as(softfloat.float64_t, @bitCast(a_f64));
    const b_sf64 = @as(softfloat.float64_t, @bitCast(b_f64));

    // Pass 'false' for signaling NaN handling
    const result = cmp_func(a_sf64, b_sf64);
    updateFFlagsFromSoftfloat(memory);

    memory.registers[rd] = @intFromBool(result); // Use @intFromBool
}

// Helper for single-precision to integer conversions using softfloat
fn doSoftFloatConvertStoI(
    comptime DestIntType: type,
    comptime SoftfloatReturnType: type,
    comptime convert_func: fn (softfloat.float32_t, u8, bool) callconv(.c) SoftfloatReturnType,
    rd: u5,
    rs1: u5,
    memory: *AddressSpace,
) !void {
    const rm = try getSoftfloatRoundingMode(memory.csr.read(Csr.FRM));
    softfloat.softfloat_roundingMode = rm;

    const a_f32 = getF32FromFreg(memory.fregs[rs1]);
    const a_sf32 = @as(softfloat.float32_t, @bitCast(a_f32));

    const result_raw = convert_func(a_sf32, rm, true);
    updateFFlagsFromSoftfloat(memory);

    // Use @intCast for potential size differences
    const result = @as(DestIntType, @intCast(result_raw));

    switch (DestIntType) {
        i32 => memory.registers[rd] = signExtend(u64, i32, signExtendI32ToU64(result)), // fcvt.w.s
        u32 => memory.registers[rd] = @as(u64, result), // fcvt.wu.s (zero-extended by default)
        i64 => memory.registers[rd] = @bitCast(result), // Use bitCast here as types match size
        u64 => memory.registers[rd] = result, // Use result directly here
        else => @compileError("Unsupported DestIntType"),
    }
}

// Helper for double-precision to integer conversions using softfloat
fn doSoftFloatConvertDtoI(
    comptime DestIntType: type,
    comptime SoftfloatReturnType: type,
    comptime convert_func: fn (softfloat.float64_t, u8, bool) callconv(.c) SoftfloatReturnType,
    rd: u5,
    rs1: u5,
    memory: *AddressSpace,
) !void {
    // Sets the rounding mode for the next operation.
    const rm = try getSoftfloatRoundingMode(memory.csr.read(Csr.FRM));
    softfloat.softfloat_roundingMode = rm;

    const a_f64 = memory.fregs[rs1];
    const a_sf64 = @as(softfloat.float64_t, @bitCast(a_f64));

    const result_raw = convert_func(a_sf64, rm, true);
    updateFFlagsFromSoftfloat(memory);

    // Use @intCast for potential size differences
    const result = @as(DestIntType, @intCast(result_raw));

    switch (DestIntType) {
        i32 => memory.registers[rd] = signExtend(u64, i32, signExtendI32ToU64(result)),
        u32 => memory.registers[rd] = @as(u64, result),
        i64 => memory.registers[rd] = @bitCast(result), // Use bitCast here as types match size
        u64 => memory.registers[rd] = result, // Use result directly here
        else => @compileError("Unsupported DestIntType"),
    }
}

// Helper for integer to single-precision conversions using softfloat
fn doSoftFloatConvertItoS(
    comptime SourceIntType: type,
    comptime convert_func: fn (SourceIntType) callconv(.c) softfloat.float32_t,
    rd: u5,
    rs1: u5,
    memory: *AddressSpace,
) !void {
    const rm = try getSoftfloatRoundingMode(memory.csr.read(Csr.FRM));
    softfloat.softfloat_roundingMode = rm;

    // Correctly extract integer value based on SourceIntType
    const a_int: SourceIntType = switch (SourceIntType) {
        i32 => @as(i32, @bitCast(@as(u32, @truncate(memory.registers[rs1])))),
        u32 => @as(u32, @truncate(memory.registers[rs1])),
        i64 => @as(i64, @bitCast(memory.registers[rs1])),
        u64 => memory.registers[rs1],
        else => @compileError("Unsupported SourceIntType"),
    };

    const result_sf32 = convert_func(a_int);
    updateFFlagsFromSoftfloat(memory);

    const result_f32: f32 = @bitCast(result_sf32);
    // NaN-box the result
    memory.fregs[rd] = @as(f64, @bitCast((@as(u64, @as(u32, @bitCast(result_f32))) | 0xFFFFFFFF00000000)));
}

// Helper for integer to double-precision conversions using softfloat
fn doSoftFloatConvertItoD(
    comptime SourceIntType: type,
    comptime convert_func: fn (SourceIntType) callconv(.c) softfloat.float64_t,
    rd: u5,
    rs1: u5,
    memory: *AddressSpace,
) !void {
    const rm = try getSoftfloatRoundingMode(memory.csr.read(Csr.FRM));
    softfloat.softfloat_roundingMode = rm;

    // Correctly extract integer value based on SourceIntType
    const a_int: SourceIntType = switch (SourceIntType) {
        i32 => @as(i32, @bitCast(@as(u32, @truncate(memory.registers[rs1])))),
        u32 => @as(u32, @truncate(memory.registers[rs1])),
        i64 => @as(i64, @bitCast(memory.registers[rs1])),
        u64 => memory.registers[rs1],
        else => @compileError("Unsupported SourceIntType"),
    };

    const result_sf64 = convert_func(a_int);
    updateFFlagsFromSoftfloat(memory);

    memory.fregs[rd] = @as(f64, @bitCast(result_sf64));
}

// Helper for ternary single-precision operations (fused multiply-add) using softfloat
fn doSoftFloatOpS_Ternary(
    rd: u5,
    rs1: u5,
    rs2: u5,
    rs3: u5,
    negate_product: bool,
    negate_addend: bool,
    memory: *AddressSpace,
) !void {
    const rm = try getSoftfloatRoundingMode(memory.csr.read(Csr.FRM));
    softfloat.softfloat_roundingMode = rm;

    const a_f32 = getF32FromFreg(memory.fregs[rs1]);
    const b_f32 = getF32FromFreg(memory.fregs[rs2]);
    const c_f32 = getF32FromFreg(memory.fregs[rs3]);

    const a_sf32 = @as(softfloat.float32_t, @bitCast(a_f32));
    const b_sf32 = @as(softfloat.float32_t, @bitCast(b_f32));
    const c_sf32 = @as(softfloat.float32_t, @bitCast(c_f32));

    // Apply sign modifications for different operations - manually negate by flipping sign bit
    const a_final = if (negate_product)
        @as(softfloat.float32_t, @bitCast(@as(u32, @bitCast(a_sf32)) ^ 0x80000000))
    else
        a_sf32;

    const c_final = if (negate_addend)
        @as(softfloat.float32_t, @bitCast(@as(u32, @bitCast(c_sf32)) ^ 0x80000000))
    else
        c_sf32;

    // Use the Berkeley SoftFloat mulAdd function
    const result_sf32 = softfloat.f32_mulAdd(a_final, b_sf32, c_final);
    updateFFlagsFromSoftfloat(memory);

    const result_f32: f32 = @bitCast(result_sf32);
    // NaN-box the result
    memory.fregs[rd] = @as(f64, @bitCast((@as(u64, @as(u32, @bitCast(result_f32))) | 0xFFFFFFFF00000000)));
}

// Helper for ternary double-precision operations (fused multiply-add) using softfloat
fn doSoftFloatOpD_Ternary(
    rd: u5,
    rs1: u5,
    rs2: u5,
    rs3: u5,
    negate_product: bool,
    negate_addend: bool,
    memory: *AddressSpace,
) !void {
    const rm = try getSoftfloatRoundingMode(memory.csr.read(Csr.FRM));
    softfloat.softfloat_roundingMode = rm;

    const a_f64 = memory.fregs[rs1];
    const b_f64 = memory.fregs[rs2];
    const c_f64 = memory.fregs[rs3];

    const a_sf64 = @as(softfloat.float64_t, @bitCast(a_f64));
    const b_sf64 = @as(softfloat.float64_t, @bitCast(b_f64));
    const c_sf64 = @as(softfloat.float64_t, @bitCast(c_f64));

    // Apply sign modifications for different operations - manually negate by flipping sign bit
    const a_final = if (negate_product)
        @as(softfloat.float64_t, @bitCast(@as(u64, @bitCast(a_sf64)) ^ 0x8000000000000000))
    else
        a_sf64;

    const c_final = if (negate_addend)
        @as(softfloat.float64_t, @bitCast(@as(u64, @bitCast(c_sf64)) ^ 0x8000000000000000))
    else
        c_sf64;

    // Use the Berkeley SoftFloat mulAdd function
    const result_sf64 = softfloat.f64_mulAdd(a_final, b_sf64, c_final);
    updateFFlagsFromSoftfloat(memory);

    memory.fregs[rd] = @as(f64, @bitCast(result_sf64));
}

fn doSoftFloatConvertDtoS(
    rd: u5,
    rs1: u5,
    memory: *AddressSpace,
) !void {
    const rm = try getSoftfloatRoundingMode(memory.csr.read(Csr.FRM));
    softfloat.softfloat_roundingMode = rm;

    const a_f64 = memory.fregs[rs1];
    const a_sf64 = @as(softfloat.float64_t, @bitCast(a_f64));

    // Use softfloat's conversion function
    const result_sf32 = softfloat.f64_to_f32(a_sf64);
    updateFFlagsFromSoftfloat(memory);

    const result_f32 = @as(f32, @bitCast(result_sf32));
    // NaN-box the result
    memory.fregs[rd] = @as(f64, @bitCast((@as(u64, @as(u32, @bitCast(result_f32))) | 0xFFFFFFFF00000000)));
}

// Helper for single-to-double precision conversions using softfloat
fn doSoftFloatConvertStoD(
    rd: u5,
    rs1: u5,
    memory: *AddressSpace,
) !void {
    // No rounding mode needed for this operation as it's exact
    // But we'll set it anyway for consistency
    const rm = try getSoftfloatRoundingMode(memory.csr.read(Csr.FRM));
    softfloat.softfloat_roundingMode = rm;

    const a_f32 = getF32FromFreg(memory.fregs[rs1]);
    const a_sf32 = @as(softfloat.float32_t, @bitCast(a_f32));

    // Use softfloat's conversion function
    const result_sf64 = softfloat.f32_to_f64(a_sf32);
    updateFFlagsFromSoftfloat(memory);

    memory.fregs[rd] = @as(f64, @bitCast(result_sf64));
}

// Helper for sign-extending i32 to u64 (manual, since signExtend can't handle it)
fn signExtendI32ToU64(val: i32) u64 {
    return @as(u64, @bitCast(@as(i64, val)));
}

pub const Csr = struct {
    // Array to hold all CSRs
    csrs: [4096]u64, // Whole thing is in one big array

    // CSR Address constants
    const CSR_SIZE: usize = 4096;
    const MXLEN: usize = 64;

    // User-level CSR addresses
    const USTATUS: u16 = 0x000; // User status register
    const UTVEC: u16 = 0x005; // User trap vector
    const UEPC: u16 = 0x041; // User exception program counter
    const UCAUSE: u16 = 0x042; // User cause
    const UTVAL: u16 = 0x043; // User bad address or instruction
    const FFLAGS: u16 = 0x001; // Floating-point exception flags
    const FRM: u16 = 0x002; // Floating-point rounding mode
    const FCSR: u16 = 0x003; // Floating-point control and status register
    const TIME: u16 = 0xc01; // Time register (Is this privileged?)

    // Floating-point exception flags
    const NX: u64 = 1 << 0; // Inexact
    const UF: u64 = 1 << 1; // Underflow
    const OF: u64 = 1 << 2; // Overflow
    const DZ: u64 = 1 << 3; // Division by Zero
    const NV: u64 = 1 << 4; // Invalid Operation

    // Supervisor-level CSR addresses
    const SSTATUS: u16 = 0x100;
    const SEDELEG: u16 = 0x102;
    const SIDELEG: u16 = 0x103;
    const SIE: u16 = 0x104;
    const STVEC: u16 = 0x105;
    const SSCRATCH: u16 = 0x140;
    const SEPC: u16 = 0x141;
    const SCAUSE: u16 = 0x142;
    const STVAL: u16 = 0x143;
    const SIP: u16 = 0x144;
    const SATP: u16 = 0x180;

    // SSTATUS field masks
    const SSTATUS_SIE_MASK: u64 = 0x2; // sstatus[1]
    const SSTATUS_SPIE_MASK: u64 = 0x20; // sstatus[5]
    const SSTATUS_UBE_MASK: u64 = 0x40; // sstatus[6]
    const SSTATUS_SPP_MASK: u64 = 0x100; // sstatus[8]
    const SSTATUS_FS_MASK: u64 = 0x6000; // sstatus[14:13]
    const SSTATUS_XS_MASK: u64 = 0x18000; // sstatus[16:15]
    const SSTATUS_SUM_MASK: u64 = 0x40000; // sstatus[18]
    const SSTATUS_MXR_MASK: u64 = 0x80000; // sstatus[19]
    const SSTATUS_UXL_MASK: u64 = 0x300000000; // sstatus[33:32]
    const SSTATUS_SD_MASK: u64 = 0x8000000000000000; // sstatus[63]
    const SSTATUS_MASK: u64 = SSTATUS_SIE_MASK | SSTATUS_SPIE_MASK | SSTATUS_UBE_MASK | SSTATUS_SPP_MASK | SSTATUS_FS_MASK | SSTATUS_XS_MASK | SSTATUS_SUM_MASK | SSTATUS_MXR_MASK | SSTATUS_UXL_MASK | SSTATUS_SD_MASK;

    // XSTATUS field ranges
    const XSTATUS_SIE = .{ .start = 1, .end = 1 };
    const XSTATUS_SPIE = .{ .start = 5, .end = 5 };
    const XSTATUS_SPP = .{ .start = 8, .end = 8 };

    // Machine-level CSR addresses
    pub const MVENDORID: u16 = 0xf11;
    pub const MARCHID: u16 = 0xf12;
    pub const MIMPID: u16 = 0xf13;
    pub const MHARTID: u16 = 0xf14;
    pub const MSTATUS: u16 = 0x300;
    pub const MISA: u16 = 0x301;
    pub const MEDELEG: u16 = 0x302;
    pub const MIDELEG: u16 = 0x303;
    pub const MIE: u16 = 0x304;
    pub const MTVEC: u16 = 0x305;
    pub const MCOUNTEREN: u16 = 0x306;
    pub const MSCRATCH: u16 = 0x340;
    pub const MEPC: u16 = 0x341;
    pub const MCAUSE: u16 = 0x342;
    pub const MTVAL: u16 = 0x343;
    pub const MIP: u16 = 0x344;
    pub const PMPCFG0: u16 = 0x3a0;
    pub const PMPADDR0: u16 = 0x3b0;

    // MSTATUS field ranges
    const MSTATUS_MIE = .{ .start = 3, .end = 3 };
    const MSTATUS_MPIE = .{ .start = 7, .end = 7 };
    const MSTATUS_MPP = .{ .start = 11, .end = 12 };
    const MSTATUS_MPRV = .{ .start = 17, .end = 17 };

    // MIP field bit masks
    const SSIP_BIT: u64 = 1 << 1;
    const MSIP_BIT: u64 = 1 << 3;
    const STIP_BIT: u64 = 1 << 5;
    const MTIP_BIT: u64 = 1 << 7;
    const SEIP_BIT: u64 = 1 << 9;
    const MEIP_BIT: u64 = 1 << 11;

    pub fn init() Csr {
        var csrs = [_]u64{0} ** CSR_SIZE;

        // Initialize MISA register
        const misa: u64 = (2 << 62) | // MXL[1:0]=2 (XLEN is 64)
            (1 << 20) | // Extensions[20] (User mode implemented)
            (1 << 18) | // Extensions[18] (Supervisor mode implemented)
            (1 << 12) | // Extensions[12] (Integer Multiply/Divide extension)
            (1 << 8) | // Extensions[8] (RV32I/64I/128I base ISA)
            (1 << 5) | // Extensions[5] (Single-precision floating-point extension)
            (1 << 3) | // Extensions[3] (Double-precision floating-point extension)
            (1 << 2) | // Extensions[2] (Compressed extension)
            1; // Extensions[0] (Atomic extension)
        csrs[MISA] = misa;

        return .{ .csrs = csrs };
    }

    pub fn incrementTime(self: *@This()) void {
        self.csrs[TIME] = self.csrs[TIME] +% 1;
    }

    // Read the value from the CSR
    pub fn read(self: *const @This(), addr: u16) u64 {
        return switch (addr) {
            FFLAGS => self.csrs[FFLAGS],
            FRM => self.csrs[FRM],
            FCSR => (self.csrs[FRM] << 5) | self.csrs[FFLAGS], // FCSR = (FRM << 5) | FFLAGS
            SSTATUS => self.csrs[MSTATUS] & SSTATUS_MASK,
            SIE => self.csrs[MIE] & self.csrs[MIDELEG],
            SIP => self.csrs[MIP] & self.csrs[MIDELEG],
            MVENDORID, MARCHID, MIMPID, MHARTID => self.csrs[@as(usize, addr)],
            else => self.csrs[@as(usize, addr)],
        };
    }

    // Write the value to the CSR
    pub fn write(self: *@This(), addr: u16, val: u64) void {
        switch (addr) {
            MVENDORID, MARCHID, MIMPID, MHARTID => {}, // Read-only
            SSTATUS => {
                self.csrs[MSTATUS] = (self.csrs[MSTATUS] & ~SSTATUS_MASK) | (val & SSTATUS_MASK);
            },
            SIE => {
                self.csrs[MIE] = (self.csrs[MIE] & ~self.csrs[MIDELEG]) | (val & self.csrs[MIDELEG]);
            },
            SIP => {
                const mask = SSIP_BIT & self.csrs[MIDELEG];
                self.csrs[MIP] = (self.csrs[MIP] & ~mask) | (val & mask);
            },
            // TODO: Make sure this is correct
            FFLAGS => {
                self.csrs[FFLAGS] = val & 0x1F;
                self.csrs[FCSR] = (self.csrs[FRM] << 5) | self.csrs[FFLAGS];
            },
            FRM => {
                self.csrs[FRM] = val & 0x7;
                self.csrs[FCSR] = (self.csrs[FRM] << 5) | self.csrs[FFLAGS];
            },
            FCSR => {
                self.csrs[FFLAGS] = val & 0x1F;
                self.csrs[FRM] = (val >> 5) & 0x7;
                self.csrs[FCSR] = val & 0xFF;
            },
            else => self.csrs[@as(usize, addr)] = val,
        }
    }

    // Read a bit from the CSR
    pub fn readBit(self: *const @This(), addr: u16, bit: usize) u64 {
        if (bit >= MXLEN) {
            // TODO: raise exception
            return 0;
        }

        return if ((self.read(addr) & (@as(u64, 1) << @truncate(bit))) != 0) 1 else 0;
    }

    // Write a bit to the CSR
    pub fn writeBit(self: *@This(), addr: u16, bit: usize, val: u64) void {
        if (bit >= MXLEN) {
            // TODO: raise exception
            return;
        }

        if (val > 1) {
            // TODO: raise exception
            return;
        }

        if (val == 1) {
            self.write(addr, self.read(addr) | (@as(u64, 1) << @truncate(bit)));
        } else if (val == 0) {
            self.write(addr, self.read(addr) & ~(@as(u64, 1) << @truncate(bit)));
        }
    }

    // Read a range of bits from the CSR
    pub fn readBits(self: *const @This(), addr: u16, start: usize, end: usize) u64 {
        if ((start >= MXLEN) or (end > MXLEN) or (start >= end)) {
            // TODO: raise exception
            return 0;
        }

        // Bitmask for high bits
        var bitmask: u64 = 0;
        if (end != 64) {
            bitmask = ~@as(u64, 0) << @truncate(end);
        }

        // Shift away low bits
        return (self.read(addr) & ~bitmask) >> @truncate(start);
    }

    // Write a range of bits to the CSR
    pub fn writeBits(self: *@This(), addr: u16, start: usize, end: usize, val: u64) void {
        if ((start >= MXLEN) or (end > MXLEN) or (start >= end)) {
            // TODO: raise exception
            return;
        }

        if ((val >> (@as(u6, @truncate(end)) - @as(u6, @truncate(start)))) != 0) {
            // TODO: raise exception
            return;
        }

        const bitmask = (~@as(u64, 0) << @truncate(end)) | ~(~@as(u64, 0) << @truncate(start));
        self.write(addr, (self.read(addr) & bitmask) | (val << @truncate(start)));
    }

    // Read a field in the SSTATUS register
    pub fn readSstatus(self: *const @This(), field: anytype) u64 {
        return self.readBits(SSTATUS, field.start, field.end);
    }

    // Write a field in the SSTATUS register
    pub fn writeSstatus(self: *@This(), field: anytype, val: u64) void {
        self.writeBits(SSTATUS, field.start, field.end, val);
    }

    // Read a field in the MSTATUS register
    pub fn readMstatus(self: *const @This(), field: anytype) u64 {
        return self.readBits(MSTATUS, field.start, field.end);
    }

    // Write a field in the MSTATUS register
    pub fn writeMstatus(self: *@This(), field: anytype, val: u64) void {
        self.writeBits(MSTATUS, field.start, field.end, val);
    }

    // Reset all CSRs
    pub fn reset(self: *@This()) void {
        self.csrs = [_]u64{0} ** CSR_SIZE;

        // Initialize MISA register
        const misa: u64 = (2 << 62) | // MXL[1:0]=2 (XLEN is 64)
            (1 << 18) | // Extensions[18] (Supervisor mode implemented)
            (1 << 12) | // Extensions[12] (Integer Multiply/Divide extension)
            (1 << 8) | // Extensions[8] (RV32I/64I/128I base ISA)
            (1 << 5) | // Extensions[5] (Single-precision floating-point extension)
            (1 << 3) | // Extensions[3] (Double-precision floating-point extension)
            (1 << 2) | // Extensions[2] (Compressed extension)
            1; // Extensions[0] (Atomic extension)
        self.csrs[MISA] = misa;
    }

    pub fn setInexactFlag(self: *@This()) void {
        self.csrs[FFLAGS] |= NX;
    }

    // Method to clear the inexact flag in fflags
    pub fn clearInexactFlag(self: *@This()) void {
        self.csrs[FFLAGS] &= ~NX;
    }

    // Method to check if the inexact flag is set
    pub fn isInexactFlagSet(self: *const @This()) bool {
        return (self.csrs[FFLAGS] & NX) != 0;
    }
};

// Privilege modes
pub const Mode = enum(u2) {
    User = 0,
    Supervisor = 1,
    Machine = 3,
    Debug = 2, // Not actually part of RISC-V spec, used for error handling
};

// Update the CSR write handler to detect and handle SATP register updates
pub fn handleCsrWrite(cpu: *RiscVCpu, csrAddr: u16, value: u64) !void {
    // Write to the CSR as usual
    cpu.csr.write(csrAddr, value);

    // Check if this is the SATP register
    if (csrAddr == Csr.SATP) {
        // When SATP is updated, we need to update the memory mode
        try vm_integration.handleSatpUpdate(cpu, value);
    }
}

// On reservation addr invalidation:
// What the RISC-V spec actually says:
// The spec is somewhat ambiguous about when LR/SC reservations must be invalidated
// It says reservations "may be invalidated" by various events, but doesn't mandate fence instructions to do so
// The spec primarily focuses on invalidation due to memory writes from other harts
// What this implementation does:
// It conservatively invalidates all reservations on fence instructions
// This is a valid implementation choice, but it's more aggressive than strictly required
// The implementation can invalidate on fence instructions, but this is an implementation choice rather than a strict requirement.
// You can do it in more places if you want to be conservative.

// Global reservation tracking system for LR/SC across all harts
const GlobalReservationTracker = struct {
    const Self = @This();
    const MAX_HARTS = 64; // Maximum number of harts supported

    // Simple reservation storage - just the address (0 means no reservation)
    reservations: [MAX_HARTS]AtomicU64,
    next_hart_id: std.atomic.Value(u32),

    fn init() Self {
        var tracker = Self{
            .reservations = undefined,
            .next_hart_id = std.atomic.Value(u32).init(0),
        };

        // Initialize all reservations to empty
        for (&tracker.reservations) |*reservation| {
            reservation.* = AtomicU64.init(0);
        }

        return tracker;
    }

    // Allocate a unique hart ID for a new CPU
    fn allocateHartId(self: *Self) u32 {
        return self.next_hart_id.fetchAdd(1, .monotonic);
    }

    // Set a reservation for a hart
    fn setReservation(self: *Self, hart_id: u32, addr: u64) void {
        if (hart_id >= MAX_HARTS) return;

        // Clear any existing reservation for this hart first
        self.clearReservation(hart_id);

        // Set the new reservation
        self.reservations[hart_id].store(addr, .release);
    }

    // Clear a specific hart's reservation
    fn clearReservation(self: *Self, hart_id: u32) void {
        if (hart_id >= MAX_HARTS) return;
        self.reservations[hart_id].store(0, .release);
    }

    // Check if a hart has a reservation for a specific address
    fn hasReservation(self: *Self, hart_id: u32, addr: u64) bool {
        if (hart_id >= MAX_HARTS) return false;
        return self.reservations[hart_id].load(.acquire) == addr;
    }

    // Invalidate all reservations that overlap with the given address range
    // This is called on every memory write to ensure LR/SC semantics
    pub fn invalidateOverlapping(self: *Self, addr: u64, size: u64) void {
        const write_start = addr;
        // Prevent undefined behaviour on overflow.  If the addition would
        // overflow the u64 range, treat the end of the write as the maximum
        // addressable value.
        const write_end = blk: {
            const res = @addWithOverflow(addr, size);
            if (res[1] == 1) {
                break :blk std.math.maxInt(u64);
            } else {
                break :blk res[0];
            }
        };

        // Check all hart reservations for overlaps
        for (&self.reservations) |*reservation| {
            const reserved_addr = reservation.load(.acquire);
            if (reserved_addr == 0) continue; // No reservation

            // For LR/SC, we need to invalidate if there's any overlap
            // RISC-V spec: "The SC must fail if a write from some other device
            // to the bytes accessed by the LR can be observed to occur between the LR and SC"
            // We conservatively invalidate on any address overlap within a reasonable range
            const INVALIDATION_GRANULARITY = 64; // Conservative granularity for invalidation
            const reservation_start = reserved_addr & ~@as(u64, INVALIDATION_GRANULARITY - 1);
            const reservation_end = reservation_start + INVALIDATION_GRANULARITY;

            // Check for overlap
            if (write_start < reservation_end and write_end > reservation_start) {
                // Invalidate this reservation
                reservation.store(0, .release);
            }
        }
    }

    // Clear all reservations (used for fence instructions, context switches, etc.)
    fn clearAllReservations(self: *Self) void {
        for (&self.reservations) |*reservation| {
            reservation.store(0, .release);
        }
    }
};
