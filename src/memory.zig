// RISC-V Virtual Memory Implementation
// Supports Sv39 and Sv48 paging modes as defined in the RISC-V privileged specification

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.array_list.Managed;
const builtin = @import("builtin");

const AtomicU64 = @import("wasm.zig").AtomicU64;
const AtomicUsize = @import("wasm.zig").AtomicUsize;
const Csr = @import("riscv.zig").Csr;
const Mode = @import("riscv.zig").Mode;
const Mutex = @import("wasm.zig").Mutex;
const print = @import("wasm.zig").print;

const WORD_LOCK_COUNT: u32 = 1024; // Power of two
var u64_word_locks: [WORD_LOCK_COUNT]Mutex = [_]Mutex{.{}} ** WORD_LOCK_COUNT;

inline fn getU64Lock(addr: usize) *Mutex {
    const word_index = addr >> 3; // 8-byte words
    const hash = @as(u32, @intCast(word_index ^ (word_index >> 10))) & (WORD_LOCK_COUNT - 1);
    return &u64_word_locks[hash];
}

// Page sizes
pub const PAGE_SIZE: u64 = 4096; // 4KiB pages
pub const MEGA_PAGE_SIZE: u64 = 2 * 1024 * 1024; // 2MiB pages
pub const GIGA_PAGE_SIZE: u64 = 1024 * 1024 * 1024; // 1GiB pages
pub const TERA_PAGE_SIZE: u64 = 1024 * 1024 * 1024 * 1024; // 1TiB pages

// Page table entry flags (RISC-V specification)
pub const PTE_V: u64 = 1 << 0; // Valid
pub const PTE_R: u64 = 1 << 1; // Readable
pub const PTE_W: u64 = 1 << 2; // Writable
pub const PTE_X: u64 = 1 << 3; // Executable
pub const PTE_U: u64 = 1 << 4; // User mode accessible
pub const PTE_G: u64 = 1 << 5; // Global mapping
pub const PTE_A: u64 = 1 << 6; // Accessed
pub const PTE_D: u64 = 1 << 7; // Dirty
pub const PTE_COW: u64 = 1 << 8; // Copy-on-Write flag (custom extension)

// Permission constants (Unicorn-compatible)
pub const UC_PROT_NONE: u64 = 0;
pub const UC_PROT_READ: u64 = PTE_R | PTE_U;
pub const UC_PROT_WRITE: u64 = PTE_W | PTE_U;
pub const UC_PROT_EXEC: u64 = PTE_X | PTE_U;
pub const UC_PROT_READ_WRITE: u64 = PTE_R | PTE_W | PTE_U;
pub const UC_PROT_READ_EXEC: u64 = PTE_R | PTE_X | PTE_U;
pub const UC_PROT_WRITE_EXEC: u64 = PTE_W | PTE_X | PTE_U;
pub const UC_PROT_ALL: u64 = PTE_R | PTE_W | PTE_X | PTE_U;
pub const UC_PROT_ALL_VALID: u64 = PTE_V | PTE_R | PTE_W | PTE_X | PTE_U; // Added by me

// =============================================================================
// ENUMS AND ERROR TYPES
// =============================================================================

// RISC-V virtual memory modes
pub const MemoryMode = enum {
    Bare, // No translation
    Sv39, // 39-bit virtual addresses (3-level page tables)
    Sv48, // 48-bit virtual addresses (4-level page tables)
};

// Memory access types for permission checking
pub const AccessType = enum {
    Read,
    Write,
    Execute,
};

// Page fault reasons
pub const PageFaultReason = error{
    InvalidAddress,
    PermissionDenied,
    NotMapped,
    OutOfMemory,
};

// ECALL hook function type
pub const EcallHookFn = *const fn (
    address_space: *AddressSpace,
    syscall_num: u64,
    args: [6]u64,
    user_data: ?*anyopaque,
) u64;

// Page-fault hook type: called after the page has been allocated / CoW handled
pub const PageFaultHookFn = *const fn (
    address_space: *AddressSpace,
    vaddr: u64,
    access_type: AccessType,
    user_data: ?*anyopaque,
) void;

// Physical memory page representation with reference counting
pub const PhysicalPage = struct {
    data: *[PAGE_SIZE]u8,
    refcount: std.atomic.Value(u32), // Reference count for shared pages

    pub fn init(allocator: Allocator) !*PhysicalPage {
        const page = try allocator.create(PhysicalPage);
        page.* = .{
            .data = try allocator.create([PAGE_SIZE]u8),
            .refcount = std.atomic.Value(u32).init(1),
        };
        // Zero the memory
        @memset(page.data[0..], 0);
        return page;
    }

    pub fn deinit(self: *PhysicalPage, allocator: Allocator) void {
        allocator.destroy(self.data);
        allocator.destroy(self);
    }

    pub fn addRef(self: *PhysicalPage) void {
        _ = self.refcount.fetchAdd(1, .acquire);
    }

    pub fn release(self: *PhysicalPage, manager: *MemoryManager) bool {
        const old_count = self.refcount.fetchSub(1, .release);
        if (old_count == 1) {
            // We were the last reference, return to free page cache
            manager.freePage(self);
            return true;
        }
        return false;
    }
};

// A page table entry
pub const PageTableEntry = struct {
    raw: if (@import("builtin").target.cpu.arch == .wasm32) std.atomic.Value(u32) else std.atomic.Value(u64),

    const RawType = if (@import("builtin").target.cpu.arch == .wasm32) u32 else u64;

    pub fn init(value: RawType) PageTableEntry {
        if (@import("builtin").target.cpu.arch == .wasm32) {
            return PageTableEntry{
                .raw = std.atomic.Value(u32).init(@intCast(value)),
            };
        } else {
            return PageTableEntry{
                .raw = std.atomic.Value(u64).init(value),
            };
        }
    }

    pub fn isValid(self: *const PageTableEntry) bool {
        return (self.raw.load(.acquire) & @as(RawType, PTE_V)) != 0;
    }

    pub fn isLeaf(self: *const PageTableEntry) bool {
        return (self.raw.load(.acquire) & @as(RawType, (PTE_R | PTE_W | PTE_X))) != 0;
    }

    pub fn isCopyOnWrite(self: *const PageTableEntry) bool {
        return (self.raw.load(.acquire) & @as(RawType, PTE_COW)) != 0;
    }

    pub fn getPhysicalAddress(self: *const PageTableEntry) u64 {
        const raw_val = self.raw.load(.acquire);
        return (@as(u64, raw_val) >> 10) << 12;
    }

    pub fn hasPermission(self: *const PageTableEntry, access_type: AccessType, user_mode: bool) bool {
        if (!self.isValid()) return false;

        const raw_value = self.raw.load(.acquire);

        // Check user mode access
        if (user_mode and (raw_value & @as(RawType, PTE_U)) == 0) {
            return false;
        }

        // Special handling for Copy-on-Write pages
        if ((raw_value & @as(RawType, PTE_COW)) != 0 and access_type == .Write) {
            return false; // Writing to CoW pages triggers page fault
        }

        // Check permission based on access type
        return switch (access_type) {
            .Read => (raw_value & @as(RawType, PTE_R)) != 0,
            .Write => (raw_value & @as(RawType, PTE_W)) != 0,
            .Execute => (raw_value & @as(RawType, PTE_X)) != 0,
        };
    }

    pub fn setFlag(self: *PageTableEntry, flag: u64, value: bool) void {
        const flag_raw = @as(RawType, @intCast(flag));
        if (value) {
            _ = self.raw.fetchOr(flag_raw, .acq_rel);
        } else {
            _ = self.raw.fetchAnd(~flag_raw, .acq_rel);
        }
    }

    // TODO: Look at this some more
    pub fn setAddress(self: *PageTableEntry, addr: u64) void {
        // Clear existing PPN and set the new one atomically
        if (@import("builtin").target.cpu.arch == .wasm32) {
            // For 32-bit PTE, we need to handle address truncation
            const new_ppn = @as(u32, @intCast((addr >> 12))) << 10;
            while (true) {
                const current = self.raw.load(.acquire);
                const new_value = (current & 0x3FF) | new_ppn;
                if (self.raw.cmpxchgWeak(current, new_value, .acq_rel, .acquire)) |_| {
                    // CAS failed, retry
                    continue;
                } else {
                    // CAS succeeded
                    break;
                }
            }
        } else {
            const new_ppn = (addr >> 12) << 10;
            while (true) {
                const current = self.raw.load(.acquire);
                const new_value = (current & 0x3FF) | new_ppn;
                if (self.raw.cmpxchgWeak(current, new_value, .acq_rel, .acquire)) |_| {
                    // CAS failed, retry
                    continue;
                } else {
                    // CAS succeeded
                    break;
                }
            }
        }
    }

    pub fn store(self: *PageTableEntry, value: u64) void {
        if (@import("builtin").target.cpu.arch == .wasm32) {
            self.raw.store(@intCast(value), .release);
        } else {
            self.raw.store(value, .release);
        }
    }

    pub fn load(self: *const PageTableEntry) u64 {
        const raw_val = self.raw.load(.acquire);
        if (@import("builtin").target.cpu.arch == .wasm32) {
            return @as(u64, raw_val);
        } else {
            return raw_val;
        }
    }
};

// Page table structure with 512 entries (RISC-V standard)
pub const PageTable = struct {
    entries: [512]PageTableEntry,
    physical_pages: [512]?*PhysicalPage, // For leaf entries
    child_tables: [512]?*PageTable, // For non-leaf entries
    allocator: Allocator,
    refcount: AtomicUsize,

    pub fn init(allocator: Allocator) !*PageTable {
        const table = try allocator.create(PageTable);
        table.* = .{
            .entries = [_]PageTableEntry{PageTableEntry.init(0)} ** 512,
            .physical_pages = [_]?*PhysicalPage{null} ** 512,
            .child_tables = [_]?*PageTable{null} ** 512,
            .allocator = allocator,
            .refcount = AtomicUsize.init(1),
        };
        return table;
    }

    pub fn deinit(self: *PageTable, manager: *MemoryManager) void {
        // Deallocate physical pages
        for (self.physical_pages) |page| {
            if (page) |p| {
                _ = p.release(manager);
            }
        }

        // Deallocate child page tables
        for (self.child_tables) |child| {
            if (child) |c| {
                c.deinit(manager);
            }
        }

        self.allocator.destroy(self);
    }

    // Get or create a child page table at the specified index
    pub fn getOrCreateChildTable(self: *PageTable, index: usize) !*PageTable {
        if (self.child_tables[index]) |child| {
            return child;
        }

        const child = try PageTable.init(self.allocator);
        self.child_tables[index] = child;

        // Update the PTE to point to the child table
        self.entries[index].store(PTE_V); // Valid but not readable/writable/executable
        self.entries[index].setAddress(@intFromPtr(&child.entries));

        return child;
    }

    // Map a virtual address to a physical page
    pub fn mapPage(
        self: *PageTable,
        vaddr: u64,
        page: *PhysicalPage,
        perm: u64,
        level: u32,
        max_level: u32,
        mm: *MemoryManager,
    ) !void {
        const vpn = [3]u9{
            @intCast((vaddr >> 12) & 0x1FF), // Level 0 VPN
            @intCast((vaddr >> 21) & 0x1FF), // Level 1 VPN
            @intCast((vaddr >> 30) & 0x1FF), // Level 2 VPN
        };

        var current_level = max_level;
        var current_table = self;

        // Traverse the page table to the correct level
        while (current_level > level) {
            const idx = vpn[current_level];
            current_table = try current_table.getOrCreateChildTable(idx);
            current_level -= 1;
        }

        // At the correct level, create a leaf entry
        const idx = vpn[level];

        // If there's already a mapping, handle accordingly
        if (current_table.entries[idx].isValid()) {
            if (current_table.physical_pages[idx]) |old_page| {
                _ = old_page.release(mm);
            }
        }

        // Create the leaf PTE with appropriate permissions
        // Include PTE_A and PTE_D to avoid page faults on access/dirty bit updates
        current_table.entries[idx].store(PTE_V | perm | PTE_A | PTE_D);
        current_table.entries[idx].setAddress(@intFromPtr(page.data));

        // Take ownership of the physical page
        current_table.physical_pages[idx] = page;
    }

    // Unmap a virtual address
    pub fn unmapPage(self: *PageTable, vaddr: u64, level: u32, max_level: u32, mm: *MemoryManager) bool {
        const vpn = [3]u9{
            @intCast((vaddr >> 12) & 0x1FF), // Level 0 VPN
            @intCast((vaddr >> 21) & 0x1FF), // Level 1 VPN
            @intCast((vaddr >> 30) & 0x1FF), // Level 2 VPN
        };

        var current_level = max_level;
        var current_table = self;

        // Traverse the page table to the correct level
        while (current_level > level) {
            const idx = vpn[current_level];

            if (!current_table.entries[idx].isValid() or current_table.entries[idx].isLeaf()) {
                return false; // Not mapped
            }

            current_table = current_table.child_tables[idx] orelse return false;
            current_level -= 1;
        }

        // At the leaf level, check if the page is mapped
        const idx = vpn[level];
        if (!current_table.entries[idx].isValid()) {
            return false;
        }

        // Release the physical page
        if (current_table.physical_pages[idx]) |page| {
            _ = page.release(mm);
            current_table.physical_pages[idx] = null;
        }

        // Invalidate the entry
        current_table.entries[idx].store(0);

        return true;
    }

    // Walk the page table to translate a virtual address
    pub fn translate(
        self: *PageTable,
        vaddr: u64,
        access: AccessType,
        user_mode: bool,
        max_level: u32,
    ) PageFaultReason!u64 {
        const vpn = [3]u9{
            @intCast((vaddr >> 12) & 0x1FF), // Level 0 VPN
            @intCast((vaddr >> 21) & 0x1FF), // Level 1 VPN
            @intCast((vaddr >> 30) & 0x1FF), // Level 2 VPN
        };

        var current_level: i32 = @intCast(max_level);
        var current_table = self;

        while (current_level >= 0) {
            const idx = vpn[@intCast(current_level)];
            const pte = &current_table.entries[idx];

            // If this gets called it really does mean is looking at memory that is not mapped (When its mapped it gets a PTE_V)
            if (!pte.isValid()) {
                return PageFaultReason.NotMapped;
            }

            if (pte.isLeaf()) {
                // Check permissions
                if (!pte.hasPermission(access, user_mode)) {
                    return PageFaultReason.PermissionDenied;
                }

                // Calculate physical address
                var phys_addr = pte.getPhysicalAddress();

                // Handle different page sizes based on level
                const level_bits: u6 = switch (current_level) {
                    0 => 12, // 4KiB page (12 bits offset)
                    1 => 21, // 2MiB page (21 bits offset)
                    2 => 30, // 1GiB page (30 bits offset)
                    else => return PageFaultReason.InvalidAddress,
                };

                // Combine physical page address with the page offset
                phys_addr |= vaddr & ((@as(u64, 1) << level_bits) - 1);

                // Update accessed and dirty bits if needed
                if (access == .Write) {
                    pte.setFlag(PTE_D, true);
                }
                pte.setFlag(PTE_A, true);

                return phys_addr;
            }

            // Move to the next level
            current_table = current_table.child_tables[idx] orelse return PageFaultReason.NotMapped;
            current_level -= 1;
        }

        return PageFaultReason.NotMapped;
    }

    // Clone a page table for Copy-on-Write semantics
    pub fn cloneTable(self: *PageTable, mm: *MemoryManager) !*PageTable {
        const clone = try PageTable.init(self.allocator);

        // Copy all entries
        for (0..512) |i| {
            if (self.entries[i].isValid()) {
                if (self.entries[i].isLeaf()) {
                    // This is a leaf entry (points to actual data)
                    clone.entries[i].store(self.entries[i].load());

                    // For CoW, make the page read-only in both parent and child
                    if ((self.entries[i].load() & PTE_W) != 0) {
                        // Only do CoW for writable pages
                        self.entries[i].setFlag(PTE_COW, true);
                        self.entries[i].setFlag(PTE_W, false);
                        clone.entries[i].setFlag(PTE_COW, true);
                        clone.entries[i].setFlag(PTE_W, false);
                    }

                    // Share the same physical page
                    if (self.physical_pages[i]) |page| {
                        page.addRef();
                        clone.physical_pages[i] = page;
                    }
                } else {
                    // This is a directory entry (points to another page table)
                    if (self.child_tables[i]) |child| {
                        // Recursively clone the child table
                        const child_clone = try child.cloneTable(mm);
                        clone.child_tables[i] = child_clone;

                        // Copy entry and update address to point to the cloned page table
                        clone.entries[i].store(self.entries[i].load());
                        clone.entries[i].setAddress(@intFromPtr(&child_clone.entries));
                    }
                }
            }
        }

        return clone;
    }

    // Handle a write to a Copy-on-Write page
    pub fn handleCopyOnWrite(
        self: *PageTable,
        vaddr: u64,
        mm: *MemoryManager,
        level: u32,
        max_level: u32,
    ) !void {
        const vpn = [3]u9{
            @intCast((vaddr >> 12) & 0x1FF), // Level 0 VPN
            @intCast((vaddr >> 21) & 0x1FF), // Level 1 VPN
            @intCast((vaddr >> 30) & 0x1FF), // Level 2 VPN
        };

        var current_level = max_level;
        var current_table = self;

        // Traverse the page table to the correct level
        while (current_level > level) {
            const idx = vpn[current_level];
            current_table = try current_table.getOrCreateChildTable(idx);
            current_level -= 1;
        }

        // At the leaf level, handle the CoW
        const idx = vpn[level];
        if (!current_table.entries[idx].isValid() or !current_table.entries[idx].isCopyOnWrite()) {
            return; // Not a CoW page
        }

        // Get the original physical page
        const old_page = current_table.physical_pages[idx] orelse return;

        // Validate old page data
        if (@intFromPtr(old_page.data) == 0) {
            print("ERROR: old_page.data is null in handleCopyOnWrite, vaddr=0x{x}\n", .{vaddr});
            return;
        }

        // Create a new physical page
        const new_page = try mm.allocPage();

        // Copy the content from the shared page to the new page
        @memcpy(new_page.data[0..PAGE_SIZE], old_page.data[0..PAGE_SIZE]);

        // Release the reference to the old page
        _ = old_page.release(mm);

        // Update the page table entry:
        // 1. Remove CoW flag
        // 2. Restore write permission
        // 3. Point to the new physical page
        current_table.entries[idx].setFlag(PTE_COW, false);
        current_table.entries[idx].setFlag(PTE_W, true);
        current_table.entries[idx].setAddress(@intFromPtr(new_page.data));
        current_table.physical_pages[idx] = new_page;
    }
};

// Address Space for a process or thread
pub const AddressSpace = struct {
    root_table: *PageTable,
    allocator: Allocator,
    mode: MemoryMode,
    mutex: Mutex, // Protect memory management operations
    cpu_state_mutex: Mutex, // Protect CPU state access during instruction execution
    ecall_hook: ?EcallHookFn,
    page_fault_hook: ?PageFaultHookFn,
    page_fault_hook_data: ?*anyopaque,

    // Per-thread CPU state
    registers: [32]u64, // General purpose registers (x0-x31)
    fregs: [32]f64, // Floating-point registers (f0-f31)
    pc: u64, // Program counter
    csr: Csr, // Control and Status Registers
    privilege_mode: Mode,

    pub fn init(allocator: Allocator, mode: MemoryMode) !*AddressSpace {
        const addr_space = try allocator.create(AddressSpace);
        addr_space.* = .{
            .root_table = try PageTable.init(allocator),
            .allocator = allocator,
            .mode = mode,
            .mutex = .{},
            .cpu_state_mutex = .{},
            .ecall_hook = null,
            .page_fault_hook = null,
            .page_fault_hook_data = null,
            .registers = [_]u64{0} ** 32,
            .fregs = [_]f64{0} ** 32,
            .pc = 0,
            .csr = Csr.init(),
            .privilege_mode = .Machine,
        };
        return addr_space;
    }

    pub fn deinit(self: *AddressSpace, manager: *MemoryManager) void {
        const old_count = self.root_table.refcount.fetchSub(1, .release);
        if (old_count == 1) {
            self.root_table.deinit(manager);
        }
        self.allocator.destroy(self);
    }

    pub fn readMemory(
        self: *AddressSpace,
        comptime T: type,
        vaddr: u64,
        user_mode: bool,
        manager: *MemoryManager,
    ) !T {
        // Check for alignment - handle misaligned accesses properly
        const alignment = @alignOf(T);
        if (vaddr % alignment != 0) {
            // Handle misaligned read by reading byte by byte
            var result: T = 0;
            const bytes = @as([*]u8, @ptrCast(&result));

            inline for (0..@sizeOf(T)) |i| {
                const byte_addr = vaddr + i;
                const phys_addr = try self.translate(byte_addr, .Read, user_mode, manager);

                if (@import("builtin").target.cpu.arch == .wasm32 and phys_addr >= 0x1_0000_0000) {
                    print("TRUNCATING phys=0x{x}\n", .{phys_addr});
                }

                // Read a single byte with alignment 1
                const byte_ptr = @as(*align(1) const u8, @ptrFromInt(@as(usize, @intCast(phys_addr))));
                bytes[i] = byte_ptr.*;
            }

            return result;
        } else {
            // Aligned access - proceed normally
            const phys_addr = try self.translate(vaddr, .Read, user_mode, manager);

            if (@import("builtin").target.cpu.arch == .wasm32 and phys_addr >= 0x1_0000_0000) {
                print("TRUNCATING phys=0x{x}\n", .{phys_addr});
            }

            // Cast physical address to correct pointer type with proper alignment
            const ptr = @as(*align(@alignOf(T)) volatile T, @ptrFromInt(@as(usize, @intCast(phys_addr))));

            // Use an atomic load for word-sized types so host ThreadSanitizer and the
            // underlying CPU treat the operation as atomic (guest hardware guarantees
            // this as well). For larger types fall back to a plain read.
            if (@sizeOf(T) == 1 or @sizeOf(T) == 2 or @sizeOf(T) == 4) {
                // These sizes are always supported atomically on all targets.
                return @atomicLoad(T, ptr, .seq_cst);
            } else if (@sizeOf(T) == 8) {
                if (builtin.target.cpu.arch != .wasm32) {
                    return @atomicLoad(T, ptr, .seq_cst);
                } else {
                    const lock = getU64Lock(@as(usize, @intCast(phys_addr)));
                    lock.lock();
                    defer lock.unlock();
                    return ptr.*;
                }
            } else {
                return ptr.*;
            }
        }
    }

    // Write memory through virtual memory translation
    pub fn writeMemory(
        self: *AddressSpace,
        comptime T: type,
        vaddr: u64,
        value: T,
        user_mode: bool,
        manager: *MemoryManager,
    ) !void {
        // Check for alignment - handle misaligned accesses properly
        const alignment = @alignOf(T);
        if (vaddr % alignment != 0) {
            // Handle misaligned write by writing byte by byte
            const bytes = @as([*]const u8, @ptrCast(&value));

            inline for (0..@sizeOf(T)) |i| {
                const byte_addr = vaddr + i;
                // For CoW, translate() will automatically handle the CoW situation
                // by detecting the CoW flag and causing a copy to be made
                const phys_addr = try self.translate(byte_addr, .Write, user_mode, manager);

                if (@import("builtin").target.cpu.arch == .wasm32 and phys_addr >= 0x1_0000_0000) {
                    print("TRUNCATING phys=0x{x}\n", .{phys_addr});
                }

                // Write a single byte with alignment 1 atomically
                const byte_ptr = @as(*align(1) volatile u8, @ptrFromInt(@as(usize, @intCast(phys_addr))));
                @atomicStore(u8, byte_ptr, bytes[i], .seq_cst);
            }
        } else {
            // Aligned access - proceed normally
            // For CoW, translate() will handle the CoW situation automatically
            const phys_addr = try self.translate(vaddr, .Write, user_mode, manager);

            if (@import("builtin").target.cpu.arch == .wasm32 and phys_addr >= 0x1_0000_0000) {
                print("TRUNCATING phys=0x{x}\n", .{phys_addr});
            }

            // Cast physical address to correct pointer type with proper alignment
            const ptr = @as(*align(@alignOf(T)) volatile T, @ptrFromInt(@as(usize, @intCast(phys_addr))));

            if (@sizeOf(T) == 1 or @sizeOf(T) == 2 or @sizeOf(T) == 4) {
                @atomicStore(T, ptr, value, .seq_cst);
            } else if (@sizeOf(T) == 8) {
                if (builtin.target.cpu.arch != .wasm32) {
                    @atomicStore(T, ptr, value, .seq_cst);
                } else {
                    const lock = getU64Lock(@as(usize, @intCast(phys_addr)));
                    lock.lock();
                    defer lock.unlock();
                    ptr.* = value;
                }
            } else {
                ptr.* = value;
            }
        }
    }

    // Allocate and map a new page
    pub fn allocPage(self: *AddressSpace, vaddr: u64, perm: u64, manager: *MemoryManager) !void {
        const page = try manager.allocPage();
        try self.mapMemory(vaddr, page, perm, manager);
    }

    // Map a region of memory
    pub fn mapMemoryRegion(
        space: *AddressSpace,
        addr: u64,
        size: u64,
        perm: u64,
        manager: *MemoryManager,
    ) !void {

        // Calculate number of pages needed
        const page_size: u64 = PAGE_SIZE;
        const start_addr = addr & ~(page_size - 1); // Align down to page boundary
        const end_addr = (addr + size + page_size - 1) & ~(page_size - 1); // Align up to page boundary
        var current_addr = start_addr;

        while (current_addr < end_addr) : (current_addr += page_size) {
            // Allocate and map each page in the region
            const page = try manager.allocPage();
            try space.mapMemory(current_addr, page, perm, manager);
        }
    }

    // Map a virtual address to a physical page
    pub fn mapMemory(self: *AddressSpace, vaddr: u64, page: *PhysicalPage, perm: u64, manager: *MemoryManager) !void {
        const max_level: u32 = switch (self.mode) {
            .Bare => return, // No translation in bare mode
            .Sv39 => 2, // 3 levels (0-2)
            .Sv48 => 3, // 4 levels (0-3)
        };

        self.mutex.lock();
        defer self.mutex.unlock();

        try self.root_table.mapPage(vaddr, page, perm, 0, max_level, manager);
    }

    // Unmap a virtual address
    pub fn unmapMemory(self: *AddressSpace, vaddr: u64, manager: *MemoryManager) bool {
        const max_level: u32 = switch (self.mode) {
            .Bare => return false, // No translation in bare mode
            .Sv39 => 2, // 3 levels (0-2)
            .Sv48 => 3, // 4 levels (0-3)
        };

        self.mutex.lock();
        defer self.mutex.unlock();

        return self.root_table.unmapPage(vaddr, 0, max_level, manager);
    }

    // Translate a virtual address to a physical address
    pub fn doTranslate(self: *AddressSpace, vaddr: u64, access: AccessType, user_mode: bool) PageFaultReason!u64 {
        if (self.mode == .Bare) {
            return vaddr; // In bare mode, no translation
        }

        const max_level: u32 = switch (self.mode) {
            .Bare => unreachable,
            .Sv39 => 2, // 3 levels
            .Sv48 => 3, // 4 levels
        };

        // Validate virtual address format
        if (self.mode == .Sv39) {
            // For Sv39, bits [63:39] must all match bit 38
            const high_bits = (vaddr >> 38) & 0x1FFFFFF;
            if (high_bits != 0 and high_bits != 0x1FFFFFF) {
                return PageFaultReason.InvalidAddress;
            }
        }

        if (self.mode == .Sv48) {
            // For Sv48, bits [63:48] must all match bit 47
            const high_bits = (vaddr >> 47) & 0xFFFF;
            if (high_bits != 0 and high_bits != 0xFFFF) {
                return PageFaultReason.InvalidAddress;
            }
        }

        self.mutex.lock();
        defer self.mutex.unlock();

        return self.root_table.translate(vaddr, access, user_mode, max_level);
    }

    // Translate virtual address to physical address using the specified address space
    pub fn translate(self: *AddressSpace, vaddr: u64, access: AccessType, user_mode: bool, manager: *MemoryManager) PageFaultReason!u64 {
        // Try to translate the address
        const result = self.doTranslate(vaddr, access, user_mode);

        // Special handling for CoW on write access
        if (result == PageFaultReason.PermissionDenied and access == .Write) {
            // Check if this is a CoW page
            const max_level: u32 = switch (self.mode) {
                .Bare => unreachable,
                .Sv39 => 2,
                .Sv48 => 3,
            };

            const vpn = [3]u9{
                @intCast((vaddr >> 12) & 0x1FF),
                @intCast((vaddr >> 21) & 0x1FF),
                @intCast((vaddr >> 30) & 0x1FF),
            };

            var current_level: i32 = @intCast(max_level);
            var current_table = self.root_table;
            var is_cow = false;

            // Navigate to the leaf PTE to check for CoW flag
            while (current_level >= 0) {
                const idx = vpn[@intCast(current_level)];
                const pte = current_table.entries[idx];

                if (!pte.isValid()) {
                    break;
                }

                if (pte.isLeaf()) {
                    is_cow = pte.isCopyOnWrite();
                    break;
                }

                if (current_table.child_tables[idx]) |child| {
                    current_table = child;
                    current_level -= 1;
                } else {
                    break;
                }
            }

            if (is_cow) {
                // Handle the CoW situation
                try self.handleCopyOnWrite(vaddr, manager);
                // Now retry the translation
                return self.doTranslate(vaddr, access, user_mode);
            }
        }

        return result;
    }

    // Clone this address space (for fork syscall)
    pub fn clone(self: *AddressSpace, manager: *MemoryManager) !*AddressSpace {
        const new_space = try AddressSpace.init(self.allocator, self.mode);

        self.mutex.lock();
        defer self.mutex.unlock();

        // Clone the page tables with Copy-on-Write semantics
        new_space.root_table.deinit(manager);

        // Create a deep copy with CoW semantics
        new_space.root_table = try self.root_table.cloneTable(manager);

        // Copy register state
        @memcpy(&new_space.registers, &self.registers);
        @memcpy(&new_space.fregs, &self.fregs);
        new_space.pc = self.pc;

        // Copy CSR state (shallow copy is enough since it's just an array of u64)
        new_space.csr = self.csr;

        return new_space;
    }

    // Handle a write to a Copy-on-Write page
    pub fn handleCopyOnWrite(self: *AddressSpace, vaddr: u64, manager: *MemoryManager) !void {
        const max_level: u32 = switch (self.mode) {
            .Bare => return, // No translation in bare mode
            .Sv39 => 2, // 3 levels (0-2)
            .Sv48 => 3, // 4 levels (0-3)
        };

        self.mutex.lock();
        defer self.mutex.unlock();

        try self.root_table.handleCopyOnWrite(vaddr, manager, 0, max_level);
    }

    // Create a new thread that shares memory with this process
    pub fn createThread(self: *AddressSpace) !*AddressSpace {
        _ = self.root_table.refcount.fetchAdd(1, .acquire);
        const thread_space = try self.allocator.create(AddressSpace);
        thread_space.* = .{
            .ecall_hook = null,
            .page_fault_hook = null,
            .page_fault_hook_data = null,
            .root_table = self.root_table, // Share the same page tables
            .allocator = self.allocator,
            .mode = self.mode,
            .mutex = .{}, // Each address space gets its own mutex
            .cpu_state_mutex = .{}, // Each thread gets its own CPU state mutex
            .registers = [_]u64{0} ** 32, // Fresh registers for the new thread
            .fregs = [_]f64{0} ** 32, // Fresh floating point registers
            .pc = 0, // New program counter
            .csr = Csr.init(), // Fresh CSR state
            .privilege_mode = .Machine, // Probably should be User but this is fine for now
        };
        return thread_space;
    }

    pub fn setEcallHook(self: *AddressSpace, hook: EcallHookFn) void {
        self.ecall_hook = hook;
    }

    pub fn removeEcallHook(self: *AddressSpace) void {
        self.ecall_hook = null;
    }

    pub fn setPageFaultHook(self: *AddressSpace, hook: PageFaultHookFn, data: ?*anyopaque) void {
        self.page_fault_hook = hook;
        self.page_fault_hook_data = data;
    }

    pub fn removePageFaultHook(self: *AddressSpace) void {
        self.page_fault_hook = null;
        self.page_fault_hook_data = null;
    }

    // Update RWX permission bits of every mapped page in a range.
    // Does not allocate or create mappings; returns PageFault if any page is unmapped.
    pub fn setRegionPerm(self: *AddressSpace, start_addr: u64, len: u64, new_perm: u64) !void {
        const page_mask: u64 = ~(PAGE_SIZE - 1);
        const aligned_start = start_addr & page_mask;
        const aligned_end = (start_addr + len + PAGE_SIZE - 1) & page_mask;

        const max_level: u32 = switch (self.mode) {
            .Bare => return,
            .Sv39 => 2,
            .Sv48 => 3,
        };

        var addr: u64 = aligned_start;
        page_loop: while (addr < aligned_end) : (addr += PAGE_SIZE) {
            const vpn = [3]u9{
                @intCast((addr >> 12) & 0x1FF),
                @intCast((addr >> 21) & 0x1FF),
                @intCast((addr >> 30) & 0x1FF),
            };

            self.mutex.lock();
            var table = self.root_table;
            var level: i32 = @intCast(max_level);
            var leaf: ?*PageTableEntry = null;
            while (level >= 0) : (level -= 1) {
                const entry = &table.entries[vpn[@intCast(level)]];
                if (!entry.isValid()) {
                    self.mutex.unlock();
                    continue :page_loop;
                }
                if (entry.isLeaf()) {
                    leaf = entry;
                    break;
                }
                table = table.child_tables[vpn[@intCast(level)]] orelse {
                    self.mutex.unlock();
                    continue :page_loop;
                };
            }

            if (leaf == null) {
                self.mutex.unlock();
                continue :page_loop;
            }

            var raw = leaf.?.load();
            raw &= ~(PTE_R | PTE_W | PTE_X | PTE_COW);
            raw |= new_perm & (PTE_R | PTE_W | PTE_X);
            leaf.?.store(raw);
            self.mutex.unlock();
        }
    }

    // Write to user pages from a privileged context without permanently
    // changing guest-visible permissions. Implementation: temporarily set
    // the W bit, perform copy, then restore the original permissions.
    pub fn writeKernelBlock(self: *AddressSpace, dest_addr: u64, data: []const u8, manager: *MemoryManager) PageFaultReason!void {
        if (data.len == 0) return;

        const page_mask: u64 = ~(PAGE_SIZE - 1);
        const start_aligned = dest_addr & page_mask;
        const end_aligned = (dest_addr + data.len + PAGE_SIZE - 1) & page_mask;

        const num_pages = @as(usize, @intCast((end_aligned - start_aligned) / PAGE_SIZE));

        var original_perms = try self.allocator.alloc(u64, num_pages);
        defer self.allocator.free(original_perms);

        // Capture current perms and grant write.
        var idx: usize = 0;
        var page: u64 = start_aligned;
        while (page < end_aligned) : (page += PAGE_SIZE) {
            // Walk to leaf PTE to read current perms.
            const vpn = [3]u9{
                @intCast((page >> 12) & 0x1FF),
                @intCast((page >> 21) & 0x1FF),
                @intCast((page >> 30) & 0x1FF),
            };

            self.mutex.lock();
            var table = self.root_table;
            var level: i32 = switch (self.mode) {
                .Bare => 0,
                .Sv39 => 2,
                .Sv48 => 3,
            };
            var leaf: ?*PageTableEntry = null;
            while (level >= 0) : (level -= 1) {
                const entry = &table.entries[vpn[@intCast(level)]];
                if (!entry.isValid()) {
                    self.mutex.unlock();
                    return PageFaultReason.NotMapped;
                }
                if (entry.isLeaf()) {
                    leaf = entry;
                    break;
                }
                table = table.child_tables[vpn[@intCast(level)]] orelse {
                    self.mutex.unlock();
                    return PageFaultReason.NotMapped;
                };
            }
            if (leaf == null) {
                self.mutex.unlock();
                return PageFaultReason.NotMapped;
            }
            const raw = leaf.?.load();
            original_perms[idx] = raw & (PTE_R | PTE_W | PTE_X);
            if ((raw & PTE_W) == 0) {
                leaf.?.setFlag(PTE_W, true);
            }
            self.mutex.unlock();
            idx += 1;
        }

        // Perform the actual write using existing guest-aware function.
        // Copy data byte-by-byte through writeMemory which honours CoW etc.
        var off: usize = 0;
        while (off < data.len) : (off += 1) {
            try self.writeMemory(u8, dest_addr + off, data[off], false, manager);
        }

        // Restore original perms.
        idx = 0;
        page = start_aligned;
        while (page < end_aligned) : (page += PAGE_SIZE) {
            const vpn = [3]u9{
                @intCast((page >> 12) & 0x1FF),
                @intCast((page >> 21) & 0x1FF),
                @intCast((page >> 30) & 0x1FF),
            };
            self.mutex.lock();
            var table = self.root_table;
            var level: i32 = switch (self.mode) {
                .Bare => 0,
                .Sv39 => 2,
                .Sv48 => 3,
            };
            var leaf: ?*PageTableEntry = null;
            while (level >= 0) : (level -= 1) {
                var entry_ptr = &table.entries[vpn[@intCast(level)]];
                if (entry_ptr.isLeaf()) {
                    leaf = entry_ptr;
                    break;
                }
                table = table.child_tables[vpn[@intCast(level)]] orelse break;
            }
            if (leaf) |p| {
                // Clear RWX then restore saved bits
                var raw = p.load();
                raw &= ~(PTE_R | PTE_W | PTE_X);
                raw |= original_perms[idx];
                p.store(raw);
            }
            self.mutex.unlock();
            idx += 1;
        }
    }
};

// Memory Manager to handle physical memory and allocate pages
pub const MemoryManager = struct {
    allocator: Allocator,
    free_pages: ArrayList(*PhysicalPage),
    mutex: Mutex,

    pub fn init(allocator: Allocator) !*MemoryManager {
        const manager = try allocator.create(MemoryManager);
        manager.* = .{
            .allocator = allocator,
            .free_pages = ArrayList(*PhysicalPage).init(allocator),
            .mutex = .{},
        };
        return manager;
    }

    pub fn deinit(self: *MemoryManager, allocator: std.mem.Allocator) void {
        // Free all cached pages
        for (self.free_pages.items) |page| {
            page.deinit(self.allocator);
        }

        self.free_pages.deinit();
        allocator.destroy(self);
    }

    // Allocate a physical page
    // The free list allows for fast allocation of previously freed pages (As its much faster than allocating a new page)
    pub fn allocPage(self: *MemoryManager) !*PhysicalPage {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Try to reuse a previously freed page
        if (self.free_pages.items.len > 0) {
            const page = self.free_pages.pop() orelse unreachable;
            // Zero the page content
            @memset(page.data[0..], 0);
            page.refcount.store(1, .release);
            return page;
        }

        // Allocate a new page
        return PhysicalPage.init(self.allocator);
    }

    // Release a physical page
    pub fn freePage(self: *MemoryManager, page: *PhysicalPage) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const current_count = page.refcount.load(.acquire);
        if (current_count <= 1) {
            // Add to free list instead of immediately deallocating
            page.refcount.store(1, .release); // Reset refcount
            self.free_pages.append(page) catch {
                // If we can't add to free list, just deallocate
                page.deinit(self.allocator);
            };
        } else {
            // Just decrement refcount
            _ = page.refcount.fetchSub(1, .release);
        }
    }
};
