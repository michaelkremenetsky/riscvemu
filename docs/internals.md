# Internals

Notes on how the pieces fit together. This assumes you've skimmed the README.

## The CPU

`RiscVCpu` (src/riscv.zig) is deliberately thin. It holds a pointer to an
`MMUResources` (the memory glue) and a hart id, and that's about it. All the
architectural state a guest cares about — registers, PC, fregs, CSRs,
privilege mode — lives on the `AddressSpace`. That split looks odd at first
but it's what makes process switching cheap: you don't save/restore anything,
you just call `executeInstruction` with a different space.

`executeInstruction` takes the space's CPU-state mutex, fetches at `space.pc`,
and dispatches. Instructions whose low two bits aren't `11` are compressed
(16-bit) and go through `executeCompressedInstruction`; everything else goes
through `executeGeneralInstruction`. If the instruction didn't set the PC
itself (branch, jump, trap), the PC is bumped by 2 or 4. x0 is forced back to
zero after every step rather than special-casing every writeback site.

Instruction fetch goes through the same translation path as loads/stores, with
execute permission checked. A fetch that page-faults gets handed to the pager
and retried once.

### Floats

All F/D arithmetic is done with Berkeley SoftFloat rather than host floating
point. Host FPUs disagree with the RISC-V spec in small ways (NaN payloads,
flag behavior, rounding of conversions) and those differences are exactly the
kind of thing spike-diff testing catches. Using softfloat everywhere means the
emulator behaves the same on x86, ARM, and wasm. The rounding mode is loaded
from the guest's `frm` CSR before each op and exception flags are copied back
into `fflags` after.

One portability wart: softfloat's `*_to_i32`-style conversions return
`int_fast32_t`, whose width depends on the platform's C ABI. The call sites
use `ReturnTypeOf(...)` to pick up whatever translate-c says the real return
type is instead of hardcoding it.

### Atomics and LR/SC

AMO instructions go through `MMUResources.atomicRmw*` / `atomicCmpxchg*`,
which operate on host memory with host atomics after translation.

LR/SC is handled with a global reservation tracker shared by all harts
(`global_reservation_tracker`). LR records (hart, address); SC succeeds only
if the reservation is still there, and any conflicting store or AMO from
another hart kills it. If you're scheduling guest threads yourself, call
`invalidateAllReservations()` on the CPU when you context-switch, the same way
real hardware drops a reservation on a trap.

## The memory system

src/memory.zig implements the privileged-spec paging structures for real:
page tables are actual radix trees of `PageTableEntry`s, translation walks
them, and the A/D bits mean what the spec says they mean.

The main objects:

- `MemoryManager` — owns physical pages. Allocation, refcounting, and the
  free list live here. One per "machine".
- `AddressSpace` — one per guest process. Root page table, memory mode
  (`Bare`, `Sv39`, `Sv48`), and all the architectural register state.
- `MMUResources` (src/vm_integration.zig) — what the CPU actually talks to.
  Wraps a `MemoryManager`, does typed reads/writes through translation, and
  owns the page fault handler.

Translation is the standard Sv39/Sv48 walk, including superpages (2M/1G).
Permission checks honor R/W/X and the U bit against the space's current
privilege mode. The accessed bit is set on any successful translation; the
dirty bit on writes. Those exist for the usual reasons — a host kernel built
on this could do eviction/LRU with A and skip writeback of clean pages with D
— and CoW relies on the write path noticing the dirty transition.

### Page faults

When translation fails, `MMUResources.readMemory`/`writeMemory` return
`PageFault` and the caller (usually the CPU) invokes
`MMUResources.handlePageFault`. The current policy is demand paging: allocate
a fresh zero page with permissions derived from the access type, map it, fire
the space's page-fault hook if one is registered, and update
`mcause`/`mtval`/`mepc` so a guest that wants to decode the fault can.
Accesses below one page are refused so guest null derefs stay faults instead
of quietly mapping page zero.

If you want different policy (mmap-backed regions, guard pages, real
segfaults), register a page-fault hook with `setPageFaultHook` and do it
there.

### fork and threads

`AddressSpace.clone(manager)` implements fork: the page table tree is copied,
writable pages in both parent and child are downgraded to read-only and marked
with a software CoW bit, and physical page refcounts are bumped. The first
write on either side takes the CoW path in `translate()`: copy the page,
remap, restore write permission, drop the refcount.

`AddressSpace.createThread()` is the clone-with-shared-VM case: the new space
shares the parent's page tables outright, but has its own registers and PC.

Don't confuse the two — threads see each other's writes immediately, forked
processes never do.

## ecall / talking to the host

The `ecall` instruction is the whole host interface. If the space has a hook
registered (`setEcallHook`), the emulator packages a7 (syscall number) and
a0–a5 (args) and calls it; the return value is written to a0. Two sentinel
returns get special treatment:

- `-315` — the hook replaced the process image (execve). PC has already been
  pointed at the new entry, so the emulator returns without touching it.
- `-242` — "blocked". The PC is left on the ecall so the instruction re-runs
  when the thread is next scheduled, which is how blocking syscalls restart.

Without a hook, ecall just logs and continues.

## wasm32

The library builds for wasm32 with shared memory, which is where a few odd
constraints come from:

- wasm32 has no 64-bit atomics, so `wasm.zig` provides `AtomicU64` /
  `AtomicUsize` types that are real `std.atomic.Value`s on native and
  mutex-guarded values on wasm.
- Physical addresses above 4 GiB can't exist in a 32-bit address space; the
  paths that would produce them are guarded on wasm.
- Printing is an `extern "env" fn consoleLog(ptr, len)` import on wasm and
  `std.debug.print` on native. Embedders on wasm need to supply `consoleLog`.

## Testing

`zig build test` runs src/vm_test.zig: mapping and permission checks, CoW
fork behavior (including forked spaces executing divergent code), thread
spaces, and the ecall hook. In the parent project the CPU is additionally
diffed instruction-by-instruction against spike and run through the official
riscv-arch-test suite; those harnesses depend on toolchains that don't belong
in a library repo, so they didn't move here.
