// Integration of virtual memory system with RISC-V CPU

const std = @import("std");
const builtin = @import("builtin");

const memory = @import("memory.zig");
const AddressSpace = memory.AddressSpace;
const MemoryMode = memory.MemoryMode;
const PageFaultReason = memory.PageFaultReason;
const AccessType = memory.AccessType;
const MemoryManager = memory.MemoryManager;
const Mutex = @import("wasm.zig").Mutex;
const print = @import("wasm.zig").print;
const riscv = @import("riscv.zig");
const RiscVCpu = riscv.RiscVCpu;

const RawType = if (@import("builtin").target.cpu.arch == .wasm32) u32 else u64;

// Maybe can move this into riscv.zig?

// NOTE: This prob should start setting user_mode to true by default (as the RISCV writes to user mode by default)

// TODO: Refactor to fix the PermissionDenied error issue.

// Removed per-page spin-locks. We now rely on host atomic operations for correctness.

pub const MMUResourcesError = error{
    PageFault,
    AccessFault,
    AddressInvalid,
    OutOfMemory,
    PermissionDenied,
    InvalidAddress,
    NotMapped,
    AddressMisaligned,
};

// Memory resources with virtual memory support
pub const MMUResources = struct {
    memory_base: RawType, // Base of physical memory
    manager: *MemoryManager,
    mode: MemoryMode, // Current memory mode

    // Initialize MMU resources
    pub fn init(allocator: std.mem.Allocator, memory_base: u64) !*MMUResources {
        const resources = try allocator.create(MMUResources);
        resources.* = .{
            .manager = try MemoryManager.init(allocator),
            .memory_base = @intCast(memory_base),
            .mode = .Bare, // Start in bare mode
        };
        return resources;
    }

    pub fn deinit(self: *MMUResources, allocator: std.mem.Allocator) void {
        self.manager.deinit(allocator);
        allocator.destroy(self);
    }

    // Map a region of memory with specified permissions
    pub fn mapMemory(self: *MMUResources, addr: u64, size: usize, perm: u64, space: *AddressSpace) MMUResourcesError!void {
        // In virtual memory modes, map the memory region
        return space.mapMemoryRegion(addr, size, perm, self.manager) catch |err| {
            print("Error mapping memory: {s}\n", .{@errorName(err)});

            // Temporary fix for now
            return error.PermissionDenied;

            // return switch (err) {
            //     PageFaultReason.InvalidAddress => MMUResourcesError.AddressInvalid,
            //     PageFaultReason.PermissionDenied => MMUResourcesError.PermissionDenied,
            //     PageFaultReason.NotMapped => MMUResourcesError.NotMapped,
            //     PageFaultReason.OutOfMemory => MMUResourcesError.OutOfMemory,
            // };
        };
    }

    // Internal read memory function without mutex (for use within atomic operations)
    fn readMemoryInternal(self: *MMUResources, comptime T: type, addr: u64, space: *AddressSpace) MMUResourcesError!T {
        // Disallow user-space access to the NULL page so that a dereference
        // of address 0 correctly raises a segmentation fault.  Linux keeps
        // the first page unmapped (see vm.mmap_min_addr).  Prevent the lazy
        // pager from mapping it by turning the access into an immediate page
        // fault rather than servicing it.
        if (addr < memory.PAGE_SIZE) {
            return MMUResourcesError.PageFault;
        }

        // Virtual memory translation with automatic page fault handling
        return space.readMemory(T, addr, false, self.manager) catch |err| {
            switch (err) {
                PageFaultReason.NotMapped => {
                    // Handle page fault by allocating memory on demand
                    self.handlePageFault(addr, AccessType.Read, space) catch |pf_err| {
                        return switch (pf_err) {
                            else => MMUResourcesError.PageFault,
                        };
                    };
                    // Retry the read after handling the page fault
                    return space.readMemory(T, addr, false, self.manager) catch |retry_err| {
                        return switch (retry_err) {
                            PageFaultReason.InvalidAddress => MMUResourcesError.AddressInvalid,
                            PageFaultReason.PermissionDenied => MMUResourcesError.AccessFault,
                            PageFaultReason.NotMapped => MMUResourcesError.PageFault,
                            PageFaultReason.OutOfMemory => MMUResourcesError.OutOfMemory,
                        };
                    };
                },
                PageFaultReason.PermissionDenied => {
                    return MMUResourcesError.AccessFault;
                },
                PageFaultReason.InvalidAddress => return MMUResourcesError.AddressInvalid,
                PageFaultReason.OutOfMemory => return MMUResourcesError.OutOfMemory,
            }
        };
    }

    // Internal write memory function without mutex (for use within atomic operations)
    fn writeMemoryInternal(self: *MMUResources, comptime T: type, addr: u64, value: T, space: *AddressSpace) MMUResourcesError!void {
        // Same NULL-page guard as in readMemoryInternal.
        if (addr < memory.PAGE_SIZE) {
            return MMUResourcesError.PageFault;
        }

        // Virtual memory translation with automatic page fault handling
        space.writeMemory(T, addr, value, false, self.manager) catch |err| {
            switch (err) {
                PageFaultReason.NotMapped, PageFaultReason.PermissionDenied => {
                    // Lazily allocate page then retry.
                    self.handlePageFault(addr, AccessType.Write, space) catch |pf_err| {
                        return switch (pf_err) {
                            else => MMUResourcesError.PageFault,
                        };
                    };
                    // Retry the write after handling the page fault
                    space.writeMemory(T, addr, value, false, self.manager) catch |retry_err| {
                        return switch (retry_err) {
                            PageFaultReason.InvalidAddress => MMUResourcesError.AddressInvalid,
                            PageFaultReason.PermissionDenied => MMUResourcesError.AccessFault,
                            PageFaultReason.NotMapped => MMUResourcesError.PageFault,
                            PageFaultReason.OutOfMemory => MMUResourcesError.OutOfMemory,
                        };
                    };
                },
                // PageFaultReason.PermissionDenied => {
                //     return MMUResourcesError.AccessFault;
                // },
                PageFaultReason.InvalidAddress => return MMUResourcesError.AddressInvalid,
                PageFaultReason.OutOfMemory => return MMUResourcesError.OutOfMemory,
            }
        };
    }

    // Read memory through virtual memory translation
    pub fn readMemory(self: *MMUResources, comptime T: type, addr: u64, space: *AddressSpace) MMUResourcesError!T {
        // Virtual memory translation - use the same mutex as atomic operations
        return self.readMemoryInternal(T, addr, space);
    }

    // Write memory through virtual memory translation
    pub fn writeMemory(self: *MMUResources, comptime T: type, addr: u64, value: T, space: *AddressSpace) MMUResourcesError!void {
        // Invalidate any overlapping reservations before performing the write
        // This ensures LR/SC semantics across all harts
        riscv.global_reservation_tracker.invalidateOverlapping(addr, @sizeOf(T));

        // Virtual memory translation - use the same mutex as atomic operations
        try self.writeMemoryInternal(T, addr, value, space);
    }

    // --- Atomic memory operations ---
    pub fn atomicRmwU32(self: *MMUResources, comptime op: std.builtin.AtomicRmwOp, addr: u64, value: u32, aq: bool, rl: bool, space: *AddressSpace) MMUResourcesError!u32 {
        // Ensure 4-byte alignment as required by the RISC-V specification.
        if (addr & 3 != 0) {
            return MMUResourcesError.AddressMisaligned;
        }
        // Invalidate any overlapping reservations before performing the atomic operation
        riscv.global_reservation_tracker.invalidateOverlapping(addr, @sizeOf(u32));

        _ = aq;
        _ = rl;

        // Translate with demand-paging support. Retry once after servicing a fault.
        var phys_addr: u64 = undefined;
        phys_addr = space.translate(addr, .Write, false, self.manager) catch |err| switch (err) {
            PageFaultReason.NotMapped, PageFaultReason.PermissionDenied => blk: {
                // Lazily allocate page then retry.
                self.handlePageFault(addr, AccessType.Write, space) catch |pf_err| {
                    return switch (pf_err) {
                        else => MMUResourcesError.PageFault,
                    };
                };
                break :blk space.translate(addr, .Write, false, self.manager) catch |retry_err| switch (retry_err) {
                    PageFaultReason.InvalidAddress => return MMUResourcesError.AddressInvalid,
                    PageFaultReason.PermissionDenied => return MMUResourcesError.AccessFault,
                    PageFaultReason.NotMapped => return MMUResourcesError.PageFault,
                    PageFaultReason.OutOfMemory => return MMUResourcesError.OutOfMemory,
                };
            },
            // PageFaultReason.PermissionDenied => return MMUResourcesError.AccessFault,
            PageFaultReason.InvalidAddress => return MMUResourcesError.AddressInvalid,
            PageFaultReason.OutOfMemory => return MMUResourcesError.OutOfMemory,
        };

        const ptr = @as(*u32, @ptrFromInt(@as(usize, @intCast(phys_addr))));
        const old = @atomicRmw(u32, ptr, op, value, .seq_cst);
        return old;
    }

    // Separate function for unsigned min/max operations
    pub fn atomicRmwU32Unsigned(self: *MMUResources, comptime op: std.builtin.AtomicRmwOp, addr: u64, value: u32, aq: bool, rl: bool, space: *AddressSpace) MMUResourcesError!u32 {
        // Ensure 4-byte alignment.
        if (addr & 3 != 0) {
            return MMUResourcesError.AddressMisaligned;
        }
        // Invalidate any overlapping reservations before performing the atomic operation
        riscv.global_reservation_tracker.invalidateOverlapping(addr, @sizeOf(u32));

        _ = aq;
        _ = rl;

        // Implement unsigned min/max with a CAS loop with page-fault retry.
        var phys_addr: u64 = undefined;
        phys_addr = space.translate(addr, .Write, false, self.manager) catch |err| switch (err) {
            PageFaultReason.NotMapped, PageFaultReason.PermissionDenied => blk: {
                self.handlePageFault(addr, AccessType.Write, space) catch |pf_err| {
                    return switch (pf_err) {
                        else => MMUResourcesError.PageFault,
                    };
                };
                break :blk space.translate(addr, .Write, false, self.manager) catch |retry_err| switch (retry_err) {
                    PageFaultReason.InvalidAddress => return MMUResourcesError.AddressInvalid,
                    PageFaultReason.PermissionDenied => return MMUResourcesError.AccessFault,
                    PageFaultReason.NotMapped => return MMUResourcesError.PageFault,
                    PageFaultReason.OutOfMemory => return MMUResourcesError.OutOfMemory,
                };
            },
            // PageFaultReason.PermissionDenied => return MMUResourcesError.AccessFault,
            PageFaultReason.InvalidAddress => return MMUResourcesError.AddressInvalid,
            PageFaultReason.OutOfMemory => return MMUResourcesError.OutOfMemory,
        };

        const ptr = @as(*u32, @ptrFromInt(@as(usize, @intCast(phys_addr))));
        var old: u32 = @atomicLoad(u32, ptr, .acquire);
        while (true) {
            const new_val = switch (op) {
                .Max => if (old > value) old else value,
                .Min => if (old < value) old else value,
                else => @panic("atomicRmwU32Unsigned only supports Min/Max operations"),
            };
            if (@cmpxchgStrong(u32, ptr, old, new_val, .seq_cst, .seq_cst) == null) break;
            old = @atomicLoad(u32, ptr, .acquire);
        }
        return old;
    }

    pub fn atomicReadU32(self: *MMUResources, addr: u64, space: *AddressSpace) MMUResourcesError!u32 {
        // Alignment check.
        if (addr & 3 != 0) {
            return MMUResourcesError.AddressMisaligned;
        }
        // Attempt translation; on fault, lazily allocate and retry once.
        var phys_addr: u64 = undefined;
        phys_addr = space.translate(addr, .Read, false, self.manager) catch |err| switch (err) {
            PageFaultReason.NotMapped, PageFaultReason.PermissionDenied => blk: {
                self.handlePageFault(addr, AccessType.Read, space) catch |pf_err| {
                    return switch (pf_err) {
                        else => MMUResourcesError.PageFault,
                    };
                };
                break :blk space.translate(addr, .Read, false, self.manager) catch |retry_err| switch (retry_err) {
                    PageFaultReason.InvalidAddress => return MMUResourcesError.AddressInvalid,
                    PageFaultReason.PermissionDenied => return MMUResourcesError.AccessFault,
                    PageFaultReason.NotMapped => return MMUResourcesError.PageFault,
                    PageFaultReason.OutOfMemory => return MMUResourcesError.OutOfMemory,
                };
            },
            // PageFaultReason.PermissionDenied => return MMUResourcesError.AccessFault,
            PageFaultReason.InvalidAddress => return MMUResourcesError.AddressInvalid,
            PageFaultReason.OutOfMemory => return MMUResourcesError.OutOfMemory,
        };

        const ptr = @as(*u32, @ptrFromInt(@as(usize, @intCast(phys_addr))));
        return @atomicLoad(u32, ptr, .acquire);
    }

    // Parameters:
    // address: Pointer to the 32-bit memory location to modify
    // expected: The value you expect to find at that location
    // new_value: The new value to store if the comparison succeeds

    pub fn atomicCmpxchgU32(self: *MMUResources, addr: u64, expected: u32, new_value: u32, aq: bool, rl: bool, space: *AddressSpace) MMUResourcesError!?u32 {
        // Ensure natural alignment.
        if (addr & 3 != 0) {
            return MMUResourcesError.AddressMisaligned;
        }
        // Invalidate any overlapping reservations before performing the atomic operation
        riscv.global_reservation_tracker.invalidateOverlapping(addr, @sizeOf(u32));

        _ = aq;
        _ = rl;

        var phys_addr: u64 = undefined;
        phys_addr = space.translate(addr, .Write, false, self.manager) catch |err| switch (err) {
            PageFaultReason.NotMapped, PageFaultReason.PermissionDenied => blk: {
                self.handlePageFault(addr, AccessType.Write, space) catch |pf_err| {
                    return switch (pf_err) {
                        else => MMUResourcesError.PageFault,
                    };
                };
                break :blk space.translate(addr, .Write, false, self.manager) catch |retry_err| switch (retry_err) {
                    PageFaultReason.InvalidAddress => return MMUResourcesError.AddressInvalid,
                    PageFaultReason.PermissionDenied => return MMUResourcesError.AccessFault,
                    PageFaultReason.NotMapped => return MMUResourcesError.PageFault,
                    PageFaultReason.OutOfMemory => return MMUResourcesError.OutOfMemory,
                };
            },
            // PageFaultReason.PermissionDenied => return MMUResourcesError.AccessFault,
            PageFaultReason.InvalidAddress => return MMUResourcesError.AddressInvalid,
            PageFaultReason.OutOfMemory => return MMUResourcesError.OutOfMemory,
        };

        const ptr = @as(*u32, @ptrFromInt(@as(usize, @intCast(phys_addr))));
        const res = @cmpxchgStrong(u32, ptr, expected, new_value, .seq_cst, .seq_cst);
        return res;
    }

    pub fn atomicRmwU64(self: *MMUResources, comptime op: std.builtin.AtomicRmwOp, addr: u64, value: u64, aq: bool, rl: bool, space: *AddressSpace) MMUResourcesError!u64 {
        // Ensure 8-byte alignment.
        if (addr & 7 != 0) {
            return MMUResourcesError.AddressMisaligned;
        }
        // Invalidate any overlapping reservations before performing the atomic operation
        riscv.global_reservation_tracker.invalidateOverlapping(addr, @sizeOf(u64));

        _ = aq;
        _ = rl;

        if (comptime builtin.target.cpu.arch == .wasm32) {
            const lock = getU64AtomicLock(addr);
            lock.lock();
            defer lock.unlock();

            const old = try self.readMemoryInternal(u64, addr, space);
            const new = switch (op) {
                .Add => old +% value,
                .Sub => old -% value,
                .And => old & value,
                .Nand => ~(old & value),
                .Or => old | value,
                .Xor => old ^ value,
                .Max => if (@as(i64, @bitCast(old)) > @as(i64, @bitCast(value))) old else value,
                .Min => if (@as(i64, @bitCast(old)) < @as(i64, @bitCast(value))) old else value,
                .Xchg => value,
            };
            try self.writeMemoryInternal(u64, addr, new, space);
            return old;
        } else {
            var phys_addr: u64 = undefined;
            phys_addr = space.translate(addr, .Write, false, self.manager) catch |err| switch (err) {
                PageFaultReason.NotMapped, PageFaultReason.PermissionDenied => blk: {
                    self.handlePageFault(addr, AccessType.Write, space) catch |pf_err| {
                        return switch (pf_err) {
                            else => MMUResourcesError.PageFault,
                        };
                    };
                    break :blk space.translate(addr, .Write, false, self.manager) catch |retry_err| switch (retry_err) {
                        PageFaultReason.InvalidAddress => return MMUResourcesError.AddressInvalid,
                        PageFaultReason.PermissionDenied => return MMUResourcesError.AccessFault,
                        PageFaultReason.NotMapped => return MMUResourcesError.PageFault,
                        PageFaultReason.OutOfMemory => return MMUResourcesError.OutOfMemory,
                    };
                },
                // PageFaultReason.PermissionDenied => return MMUResourcesError.AccessFault,
                PageFaultReason.InvalidAddress => return MMUResourcesError.AddressInvalid,
                PageFaultReason.OutOfMemory => return MMUResourcesError.OutOfMemory,
            };

            const ptr = @as(*u64, @ptrFromInt(@as(usize, @intCast(phys_addr))));
            const old = @atomicRmw(u64, ptr, op, value, .seq_cst);
            return old;
        }
    }

    // Separate function for unsigned min/max operations
    pub fn atomicRmwU64Unsigned(self: *MMUResources, comptime op: std.builtin.AtomicRmwOp, addr: u64, value: u64, aq: bool, rl: bool, space: *AddressSpace) MMUResourcesError!u64 {
        if (addr & 7 != 0) {
            return MMUResourcesError.AddressMisaligned;
        }
        // Invalidate any overlapping reservations before performing the atomic operation
        riscv.global_reservation_tracker.invalidateOverlapping(addr, @sizeOf(u64));

        _ = aq;
        _ = rl;

        if (comptime builtin.target.cpu.arch == .wasm32) {
            const lock = getU64AtomicLock(addr);
            lock.lock();
            defer lock.unlock();

            var old: u64 = try self.readMemoryInternal(u64, addr, space);
            while (true) {
                const new_val = switch (op) {
                    .Max => if (old > value) old else value,
                    .Min => if (old < value) old else value,
                    else => @panic("atomicRmwU64Unsigned only supports Min/Max operations"),
                };
                try self.writeMemoryInternal(u64, addr, new_val, space);
                const check = try self.readMemoryInternal(u64, addr, space);
                if (check == new_val) break; // success
                old = check; // retry
            }
            return old;
        } else {
            var phys_addr: u64 = undefined;
            phys_addr = space.translate(addr, .Write, false, self.manager) catch |err| switch (err) {
                PageFaultReason.NotMapped, PageFaultReason.PermissionDenied => blk: {
                    self.handlePageFault(addr, AccessType.Write, space) catch |pf_err| {
                        return switch (pf_err) {
                            else => MMUResourcesError.PageFault,
                        };
                    };
                    break :blk space.translate(addr, .Write, false, self.manager) catch |retry_err| switch (retry_err) {
                        PageFaultReason.InvalidAddress => return MMUResourcesError.AddressInvalid,
                        PageFaultReason.PermissionDenied => return MMUResourcesError.AccessFault,
                        PageFaultReason.NotMapped => return MMUResourcesError.PageFault,
                        PageFaultReason.OutOfMemory => return MMUResourcesError.OutOfMemory,
                    };
                },
                // PageFaultReason.PermissionDenied => return MMUResourcesError.AccessFault,
                PageFaultReason.InvalidAddress => return MMUResourcesError.AddressInvalid,
                PageFaultReason.OutOfMemory => return MMUResourcesError.OutOfMemory,
            };

            const ptr = @as(*u64, @ptrFromInt(@as(usize, @intCast(phys_addr))));
            var old: u64 = @atomicLoad(u64, ptr, .acquire);
            while (true) {
                const new_val = switch (op) {
                    .Max => if (old > value) old else value,
                    .Min => if (old < value) old else value,
                    else => @panic("atomicRmwU64Unsigned only supports Min/Max operations"),
                };
                if (@cmpxchgStrong(u64, ptr, old, new_val, .seq_cst, .seq_cst) == null) break;
                old = @atomicLoad(u64, ptr, .acquire);
            }
            return old;
        }
    }

    pub fn atomicReadU64(self: *MMUResources, addr: u64, space: *AddressSpace) MMUResourcesError!u64 {
        if (addr & 7 != 0) {
            return MMUResourcesError.AddressMisaligned;
        }
        if (comptime builtin.target.cpu.arch == .wasm32) {
            const lock = getU64AtomicLock(addr);
            lock.lock();
            defer lock.unlock();
            return try self.readMemoryInternal(u64, addr, space);
        } else {
            var phys_addr: u64 = undefined;
            phys_addr = space.translate(addr, .Read, false, self.manager) catch |err| switch (err) {
                PageFaultReason.NotMapped, PageFaultReason.PermissionDenied => blk: {
                    self.handlePageFault(addr, AccessType.Read, space) catch |pf_err| {
                        return switch (pf_err) {
                            else => MMUResourcesError.PageFault,
                        };
                    };
                    break :blk space.translate(addr, .Read, false, self.manager) catch |retry_err| switch (retry_err) {
                        PageFaultReason.InvalidAddress => return MMUResourcesError.AddressInvalid,
                        PageFaultReason.PermissionDenied => return MMUResourcesError.AccessFault,
                        PageFaultReason.NotMapped => return MMUResourcesError.PageFault,
                        PageFaultReason.OutOfMemory => return MMUResourcesError.OutOfMemory,
                    };
                },
                // PageFaultReason.PermissionDenied => return MMUResourcesError.AccessFault,
                PageFaultReason.InvalidAddress => return MMUResourcesError.AddressInvalid,
                PageFaultReason.OutOfMemory => return MMUResourcesError.OutOfMemory,
            };
            const ptr = @as(*u64, @ptrFromInt(@as(usize, @intCast(phys_addr))));
            return @atomicLoad(u64, ptr, .acquire);
        }
    }

    pub fn atomicCmpxchgU64(self: *MMUResources, addr: u64, expected: u64, new_value: u64, aq: bool, rl: bool, space: *AddressSpace) MMUResourcesError!?u64 {
        if (addr & 7 != 0) {
            return MMUResourcesError.AddressMisaligned;
        }
        // Invalidate any overlapping reservations before performing the atomic operation
        riscv.global_reservation_tracker.invalidateOverlapping(addr, @sizeOf(u64));

        _ = aq;
        _ = rl;

        if (comptime builtin.target.cpu.arch == .wasm32) {
            const lock = getU64AtomicLock(addr);
            lock.lock();
            defer lock.unlock();
            const old = try self.readMemoryInternal(u64, addr, space);
            if (old == expected) {
                try self.writeMemoryInternal(u64, addr, new_value, space);
                return null;
            } else {
                return old;
            }
        } else {
            var phys_addr: u64 = undefined;
            phys_addr = space.translate(addr, .Write, false, self.manager) catch |err| switch (err) {
                PageFaultReason.NotMapped, PageFaultReason.PermissionDenied => blk: {
                    self.handlePageFault(addr, AccessType.Write, space) catch |pf_err| {
                        return switch (pf_err) {
                            else => MMUResourcesError.PageFault,
                        };
                    };
                    break :blk space.translate(addr, .Write, false, self.manager) catch |retry_err| switch (retry_err) {
                        PageFaultReason.InvalidAddress => return MMUResourcesError.AddressInvalid,
                        PageFaultReason.PermissionDenied => return MMUResourcesError.AccessFault,
                        PageFaultReason.NotMapped => return MMUResourcesError.PageFault,
                        PageFaultReason.OutOfMemory => return MMUResourcesError.OutOfMemory,
                    };
                },
                // PageFaultReason.PermissionDenied => return MMUResourcesError.AccessFault,
                PageFaultReason.InvalidAddress => return MMUResourcesError.AddressInvalid,
                PageFaultReason.OutOfMemory => return MMUResourcesError.OutOfMemory,
            };
            const ptr = @as(*u64, @ptrFromInt(@as(usize, @intCast(phys_addr))));
            const res = @cmpxchgStrong(u64, ptr, expected, new_value, .seq_cst, .seq_cst);
            return res;
        }
    }

    // Handle page fault exception
    pub fn handlePageFault(self: *MMUResources, vaddr: u64, access_type: AccessType, space: *AddressSpace) !void {
        // Prevent the pager from allocating the NULL page.  Any access to
        // addresses below one page should remain unmapped and result in a
        // fault delivered to user space.
        if (vaddr < memory.PAGE_SIZE) {
            return MMUResourcesError.AddressInvalid;
        }

        // For demonstration purposes, just map a fresh page for faults. Any copy-on-write
        // handling is now performed inside AddressSpace.translate(), so we don't replicate
        // that logic here in the pager.

        const user_flag: u64 = if (space.privilege_mode == .User) memory.PTE_U else 0;
        const perm: u64 = switch (access_type) {
            .Read => memory.PTE_R | user_flag,
            .Write => memory.PTE_R | memory.PTE_W | user_flag,
            .Execute => memory.PTE_R | memory.PTE_X | user_flag,
        };

        // Round address down to page boundary
        const page_addr = vaddr & ~@as(u64, 0xFFF);

        // Allocate and map a new page
        try space.allocPage(page_addr, perm, self.manager);

        // Call page-fault hook if registered
        if (space.page_fault_hook) |hook| {
            hook(space, vaddr, access_type, space.page_fault_hook_data);
        }

        // Update CSR registers to reflect the fault so that a Linux guest can
        // decode the exception correctly.  The cause codes follow the RISC-V
        // privileged spec (Table 3.6).
        const cause_code: u64 = switch (access_type) {
            .Execute => 12, // Instruction page fault
            .Read => 13, // Load page fault
            .Write => 15, // Store/AMO page fault
        };
        space.csr.write(riscv.Csr.MCAUSE, cause_code);
        space.csr.write(riscv.Csr.MTVAL, vaddr);
        // Resume at the faulting instruction so, once the page is present, the
        // guest re-executes it.
        space.csr.write(riscv.Csr.MEPC, space.pc);
    }
};

const WORD_LOCK_COUNT: u32 = 1024;
var u64_atomic_locks: [WORD_LOCK_COUNT]Mutex = [_]Mutex{.{}} ** WORD_LOCK_COUNT;

inline fn getU64AtomicLock(addr: u64) *Mutex {
    const word_index: u64 = addr >> 3;
    const hash = @as(u32, @intCast((word_index ^ (word_index >> 10)) & (WORD_LOCK_COUNT - 1)));
    return &u64_atomic_locks[hash];
}
