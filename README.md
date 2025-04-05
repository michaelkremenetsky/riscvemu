# riscvemu

A RISC-V (RV64) emulator library written in Zig. I pulled this out of a bigger
project of mine that runs Linux userspace in the browser, so the design leans
toward that use case: it's a library you embed, not a system emulator you boot.
Your host code plays the role of the kernel — you map memory, point the PC at
some code, and catch `ecall`s with a hook.

What it does:

- RV64 IMAFDC + Zicsr. Compressed instructions, mul/div, atomics, single and
  double precision floats.
- Floats go through [Berkeley SoftFloat](http://www.jhauser.us/arithmetic/SoftFloat.html)
  (vendored under `deps/softfloat`) rather than host floats, so rounding modes
  and fflags behave like the spec says instead of like your host CPU feels like.
- A real MMU: Sv39/Sv48 page tables, demand paging, dirty/accessed bits,
  copy-on-write `fork()`, and shared address spaces for threads.
- LR/SC reservations are tracked globally across harts, so you can run multiple
  CPUs against shared memory and atomics stay correct.
- Builds for native targets and wasm32. On wasm32 there are no 64-bit atomics,
  so those fall back to mutex-guarded values internally.

What it doesn't do: no devices, no PLIC/CLINT, no interrupt delivery, no
booting an OS. If a guest needs a kernel, you are the kernel (see the ecall
hook below).

## Using it

Add it to your `build.zig.zon`, then in `build.zig`:

```zig
const riscvemu = b.dependency("riscvemu", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.addImport("riscvemu", riscvemu.module("riscvemu"));
```

Small end-to-end example — map a page, write a few instructions, step until
`ebreak`:

```zig
const std = @import("std");
const riscvemu = @import("riscvemu");
const mem = riscvemu.memory;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var cpu = try riscvemu.RiscVCpu.init(0, allocator);
    defer cpu.deinit(allocator);

    const space = try riscvemu.AddressSpace.init(allocator, .Sv39);
    defer space.deinit(cpu.resources.manager);

    const code: u64 = 0x1000;
    try cpu.resources.mapMemory(code, 4096, mem.PTE_R | mem.PTE_W | mem.PTE_X | mem.PTE_U, space);

    try cpu.resources.writeMemory(u32, code, 0x00200513, space); // addi a0, zero, 2
    try cpu.resources.writeMemory(u32, code + 4, 0x00300593, space); // addi a1, zero, 3
    try cpu.resources.writeMemory(u32, code + 8, 0x00b50633, space); // add  a2, a0, a1
    try cpu.resources.writeMemory(u32, code + 12, 0x00100073, space); // ebreak

    space.pc = code;
    while (try cpu.resources.readMemory(u32, space.pc, space) != 0x00100073) {
        try cpu.executeInstruction(space, null);
    }

    std.debug.print("a2 = {d}\n", .{space.registers[12]}); // 5
}
```

### Handling syscalls

Register an ecall hook on an address space and every `ecall` lands in your
code with the syscall number (a7) and arguments (a0–a5) already pulled out of
the registers. Whatever you return goes back into a0:

```zig
fn handleEcall(space: *riscvemu.AddressSpace, num: u64, args: [6]u64, user_data: ?*anyopaque) u64 {
    _ = space;
    _ = user_data;
    std.debug.print("syscall {d} ({d}, {d}, ...)\n", .{ num, args[0], args[1] });
    return 0;
}

space.setEcallHook(handleEcall);
```

This is enough to run static Linux binaries if you implement the syscalls they
use. That's exactly what the parent project does.

### Processes and threads

Each guest process gets its own `AddressSpace`. For fork semantics,
`space.clone(manager)` gives you a copy-on-write copy; for threads,
`space.createThread()` gives you a new space that shares the same page tables
(registers and PC are per-space). Switching processes is just executing
against a different space — the CPU itself is stateless between steps apart
from the LR/SC reservation, which you can drop on a context switch with
`cpu.invalidateAllReservations()`.

## Building

Needs Zig 0.15. There's nothing to build for the library itself; consumers pull
it in as a module. To run the tests:

```
zig build test
```

The tests cover the MMU (mapping, permissions, CoW fork, thread spaces) and
run small instruction sequences through the CPU, including the ecall hook
path.

## Layout

- `src/riscv.zig` — the CPU. Fetch/decode/execute for the general and
  compressed instruction sets, CSRs, float ops via softfloat.
- `src/memory.zig` — page tables, address spaces, the physical page allocator,
  translation and permission checks.
- `src/vm_integration.zig` — `MMUResources`, the glue the CPU uses to read and
  write guest memory, plus the page fault handler and guest-facing atomics.
- `src/wasm.zig` — small shims (atomics, mutex, printing) that make the same
  code build on wasm32 and native.
- `deps/softfloat` — vendored Berkeley SoftFloat 3d sources.

There are more notes on the design in [docs/internals.md](docs/internals.md).

One thing to know on wasm32: `src/wasm.zig` declares an
`extern "env" fn consoleLog(ptr, len)` for printing, so your embedder needs to
provide that import (or never hit a code path that prints).

## License

MIT. The vendored SoftFloat code is BSD-3-Clause, copyright the Regents of the
University of California — see the headers in `deps/softfloat`.
