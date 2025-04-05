const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const is_wasm = target.result.cpu.arch == .wasm32;

    const softfloat_files = findCSourceFiles(b, "deps/softfloat") catch @panic("could not list deps/softfloat");

    // The public module. Consumers get this with:
    //   const zriscv = b.dependency("zriscv", .{...}).module("zriscv");
    const mod = b.addModule("zriscv", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = !is_wasm,
    });
    mod.addIncludePath(b.path("deps/softfloat"));
    mod.addCSourceFiles(.{
        .root = b.path("deps/softfloat"),
        .files = softfloat_files,
        .flags = if (is_wasm) &.{ "-nostdlib", "-ffreestanding" } else &.{},
    });

    // zig build test
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/vm_test.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = !is_wasm,
    });
    test_mod.addIncludePath(b.path("deps/softfloat"));
    test_mod.addCSourceFiles(.{
        .root = b.path("deps/softfloat"),
        .files = softfloat_files,
        .flags = if (is_wasm) &.{ "-nostdlib", "-ffreestanding" } else &.{},
    });

    const tests = b.addTest(.{ .root_module = test_mod });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run the emulator tests");
    test_step.dependOn(&run_tests.step);
}

fn findCSourceFiles(b: *std.Build, dir_path: []const u8) ![]const []const u8 {
    var sources: std.ArrayList([]const u8) = .empty;

    var dir = try std.fs.cwd().openDir(dir_path, .{ .iterate = true });
    defer dir.close();

    var it = dir.iterate();
    while (try it.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".c")) continue;
        try sources.append(b.allocator, b.dupe(entry.name));
    }

    return sources.items;
}
