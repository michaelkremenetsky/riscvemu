const std = @import("std");

const is_wasm = @import("builtin").target.cpu.arch == .wasm32;

pub extern "env" fn consoleLog(ptr: [*]const u8, len: usize) void;

pub fn print(comptime fmt: []const u8, args: anytype) void {
    if (is_wasm) {
        var buf: [1024]u8 = undefined;
        const slice = std.fmt.bufPrint(&buf, fmt, args) catch return;
        consoleLog(slice.ptr, slice.len);
    } else {
        std.debug.print(fmt, args);
    }
}

// WASM32 doesn't support 64-bit atomics, so we fallback to using mutexes

// Will be testing with this everywhere for now will switch to WASM only later.
// const use_mutex = @import("builtin").target.cpu.arch == .wasm32;
const use_mutex = @import("builtin").target.cpu.arch == .wasm32;

pub const AtomicU64 = if (use_mutex) struct {
    value: u64,
    mutex: Mutex,

    pub fn init(initial_value: u64) @This() {
        return .{
            .value = initial_value,
            .mutex = .{},
        };
    }

    pub fn load(self: *const @This(), comptime ordering: std.builtin.AtomicOrder) u64 {
        _ = ordering;
        const self_mut = @constCast(self);
        self_mut.mutex.lock();
        defer self_mut.mutex.unlock();
        return self.value;
    }

    pub fn store(self: *@This(), value: u64, comptime ordering: std.builtin.AtomicOrder) void {
        _ = ordering;
        self.mutex.lock();
        defer self.mutex.unlock();
        self.value = value;
    }

    pub fn fetchOr(self: *@This(), operand: u64, comptime ordering: std.builtin.AtomicOrder) u64 {
        _ = ordering;
        self.mutex.lock();
        defer self.mutex.unlock();
        const old_value = self.value;
        self.value |= operand;
        return old_value;
    }

    pub fn fetchAnd(self: *@This(), operand: u64, comptime ordering: std.builtin.AtomicOrder) u64 {
        _ = ordering;
        self.mutex.lock();
        defer self.mutex.unlock();
        const old_value = self.value;
        self.value &= operand;
        return old_value;
    }

    pub fn fetchAdd(self: *@This(), operand: u64, comptime ordering: std.builtin.AtomicOrder) u64 {
        _ = ordering;
        self.mutex.lock();
        defer self.mutex.unlock();
        const old_value = self.value;
        self.value += operand;
        return old_value;
    }

    pub fn fetchSub(self: *@This(), operand: u64, comptime ordering: std.builtin.AtomicOrder) u64 {
        _ = ordering;
        self.mutex.lock();
        defer self.mutex.unlock();
        const old_value = self.value;
        // Prevent unsigned underflow which triggers a safety panic in debug builds
        if (operand > self.value) {
            // Saturate at zero – this should never happen for well-behaved callers,
            // but avoids a crash if the reference counter is already 0.
            self.value = 0;
        } else {
            self.value -= operand;
        }
        return old_value;
    }

    pub fn cmpxchgWeak(
        self: *@This(),
        expected: u64,
        desired: u64,
        comptime success_order: std.builtin.AtomicOrder,
        comptime failure_order: std.builtin.AtomicOrder,
    ) ?u64 {
        _ = success_order;
        _ = failure_order;
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.value == expected) {
            self.value = desired;
            return null; // Success
        } else {
            return self.value; // Failure, return current value
        }
    }

    pub fn cmpxchgStrong(
        self: *@This(),
        expected: u64,
        desired: u64,
        comptime success_order: std.builtin.AtomicOrder,
        comptime failure_order: std.builtin.AtomicOrder,
    ) ?u64 {
        // For mutex-based implementation, strong and weak are the same
        return self.cmpxchgWeak(expected, desired, success_order, failure_order);
    }
} else std.atomic.Value(u64);

// Similar wrapper for usize (which might be 64-bit on some platforms)
pub const AtomicUsize = if (use_mutex) struct {
    value: usize,
    mutex: Mutex,

    pub fn init(initial_value: usize) @This() {
        return .{
            .value = initial_value,
            .mutex = .{},
        };
    }

    pub fn load(self: *const @This(), comptime ordering: std.builtin.AtomicOrder) usize {
        _ = ordering;
        const self_mut = @constCast(self);
        self_mut.mutex.lock();
        defer self_mut.mutex.unlock();
        return self.value;
    }

    pub fn store(self: *@This(), value: usize, comptime ordering: std.builtin.AtomicOrder) void {
        _ = ordering;
        self.mutex.lock();
        defer self.mutex.unlock();
        self.value = value;
    }

    pub fn fetchAdd(self: *@This(), operand: usize, comptime ordering: std.builtin.AtomicOrder) usize {
        _ = ordering;
        self.mutex.lock();
        defer self.mutex.unlock();
        const old_value = self.value;
        self.value += operand;
        return old_value;
    }

    pub fn fetchSub(self: *@This(), operand: usize, comptime ordering: std.builtin.AtomicOrder) usize {
        _ = ordering;
        self.mutex.lock();
        defer self.mutex.unlock();
        const old_value = self.value;
        // Prevent unsigned underflow which triggers a safety panic in debug builds
        if (operand > self.value) {
            // Saturate at zero to avoid overflow; unexpected for correct logic
            self.value = 0;
        } else {
            self.value -= operand;
        }
        return old_value;
    }

    pub fn fetchOr(self: *@This(), operand: usize, comptime ordering: std.builtin.AtomicOrder) usize {
        _ = ordering;
        self.mutex.lock();
        defer self.mutex.unlock();
        const old_value = self.value;
        self.value |= operand;
        return old_value;
    }

    pub fn fetchAnd(self: *@This(), operand: usize, comptime ordering: std.builtin.AtomicOrder) usize {
        _ = ordering;
        self.mutex.lock();
        defer self.mutex.unlock();
        const old_value = self.value;
        self.value &= operand;
        return old_value;
    }

    pub fn cmpxchgWeak(
        self: *@This(),
        expected: usize,
        desired: usize,
        comptime success_order: std.builtin.AtomicOrder,
        comptime failure_order: std.builtin.AtomicOrder,
    ) ?usize {
        _ = success_order;
        _ = failure_order;
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.value == expected) {
            self.value = desired;
            return null; // Success
        } else {
            return self.value; // Failure, return current value
        }
    }

    pub fn cmpxchgStrong(
        self: *@This(),
        expected: usize,
        desired: usize,
        comptime success_order: std.builtin.AtomicOrder,
        comptime failure_order: std.builtin.AtomicOrder,
    ) ?usize {
        // For mutex-based implementation, strong and weak are the same
        return self.cmpxchgWeak(expected, desired, success_order, failure_order);
    }
} else std.atomic.Value(usize);

pub const Mutex = if (is_wasm) WasmMutex else std.Thread.Mutex;

// WASM mutex states
const UNLOCKED: u32 = 0;
const LOCKED: u32 = 1;
const LOCKED_WITH_WAITERS: u32 = 2;

// WASM-specific mutex implementation using atomics
pub const WasmMutex = struct {
    state: std.atomic.Value(u32) align(4) = std.atomic.Value(u32).init(UNLOCKED),

    pub fn init() WasmMutex {
        return WasmMutex{};
    }

    pub fn lock(self: *WasmMutex) void {
        // Fast path: try to acquire lock directly
        // If state is UNLOCKED, set it to LOCKED
        if (self.state.cmpxchgStrong(UNLOCKED, LOCKED, .acquire, .monotonic) == null) {
            return; // Successfully acquired
        }

        // Slow path: contention exists
        while (true) {
            // Try to acquire the lock if it's available
            var current = self.state.load(.acquire);
            if (current == UNLOCKED) {
                if (self.state.cmpxchgStrong(UNLOCKED, LOCKED, .acquire, .monotonic) == null) {
                    return; // Successfully acquired
                }
                // CAS failed, state changed - retry
                continue;
            }

            // Lock is held, we need to wait
            // First, ensure the state indicates there are waiters
            if (current == LOCKED) {
                // Try to upgrade to LOCKED_WITH_WAITERS atomically
                // Use weak CAS since we're in a loop anyway
                if (self.state.cmpxchgWeak(LOCKED, LOCKED_WITH_WAITERS, .acq_rel, .acquire) == null) {
                    // Successfully upgraded, now current represents the new state
                    current = LOCKED_WITH_WAITERS;
                } else {
                    // Upgrade failed, state changed - reload and retry
                    continue;
                }
            }

            // At this point, current should be LOCKED_WITH_WAITERS
            // Wait for the lock to be released
            if (current == LOCKED_WITH_WAITERS) {
                const wait_result = wasmAtomicWait32(@ptrCast(&self.state.raw), LOCKED_WITH_WAITERS, -1);

                // Regardless of wait result, we need to try acquiring the lock again
                // The wait may have been spurious, timed out, or we were legitimately woken up
                _ = wait_result;

                // Continue to the top of the loop to try acquiring
                continue;
            }

            // Should not reach here, but continue just in case
            continue;
        }
    }

    pub fn unlock(self: *WasmMutex) void {
        // Release the lock and get the previous state
        const old_state = self.state.swap(UNLOCKED, .release);

        // Only wake waiters if there were actually waiters
        if (old_state == LOCKED_WITH_WAITERS) {
            // Wake up one waiting thread
            const woken = wasmAtomicNotify32(@ptrCast(&self.state.raw), 1);
            _ = woken; // Ignore the number of threads woken
        }
        // If old_state was LOCKED (no waiters), no need to wake anyone
        // If old_state was UNLOCKED, this is a double-unlock bug (but we handle it gracefully)
    }

    pub fn tryLock(self: *WasmMutex) bool {
        return self.state.cmpxchgWeak(UNLOCKED, LOCKED, .acquire, .monotonic) == null;
    }
};

// External WASM functions for atomic wait/notify (to be implemented in JavaScript)
extern "env" fn wasmAtomicWait32(ptr: *u32, expected: u32, timeout: i32) i32;
extern "env" fn wasmAtomicNotify32(ptr: *u32, count: u32) u32;
