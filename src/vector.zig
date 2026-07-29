// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Michael Pollind
const std = @import("std");
const Float = std.meta.Float;

const Mat = @import("matrix.zig").Mat;
const testing = @import("testing.zig");
const meta = @import("meta.zig");
const simd = @import("simd.zig");

pub const Vec4f32 = @Vector(4, f32);
pub const Vec3f32 = @Vector(3, f32);
pub const Vec2f32 = @Vector(2, f32);

pub const Vec4f64 = @Vector(4, f64);
pub const Vec3f64 = @Vector(3, f64);
pub const Vec2f64 = @Vector(2, f64);

/// Turn a length-N vector into an N x 1 column matrix, for use as the right operand of `mul`.
pub fn to_mat(vec: anytype) Mat(meta.Child(@TypeOf(vec)), 1, meta.lengthOf(@TypeOf(vec))) {
    var result: Mat(meta.Child(@TypeOf(vec)), 1, meta.lengthOf(@TypeOf(vec))) = .zero;
    inline for (0..meta.lengthOf(@TypeOf(vec))) |i| {
        result.items[0][i] = vec[i];
    }
    return result;
}

pub fn extract(vec: anytype, comptime len: usize) meta.Reshape(@TypeOf(vec), len) {
    comptime {
        std.debug.assert(len <= meta.lengthOf(@TypeOf(vec)));
    }
    var result: @Vector(len, meta.Child(@TypeOf(vec))) = undefined;
    inline for (0..len) |i| {
        result[i] = vec[i];
    }
    return result;
}

/// Returns a new vector with the components swizzled.
///
/// example:
/// ```
/// const v = Vec(2, f32).init(.{ 1, 2 });
/// const v2 = v.sw("yx"); // <= here x and y are swapped
/// ```
/// v2 is equal to `Vec(2, f32).init(.{ 2, 1 });`
///
/// this is also valid:
/// ```
/// const v = Vec(2, f32).init(.{ 1, 2, 3, 4 });
/// const v2 = v.sw("wxx");
/// ```
/// here v2 is equal to `Vec(3, f32).init(.{ 4, 1, 1 });`
pub fn swizzle(vec: anytype, comptime components: []const u8) meta.Reshape(@TypeOf(vec), components.len) {
    const T = meta.Child(@TypeOf(vec));
    const v: meta.AsVector(@TypeOf(vec)) = vec;
    comptime var mask: [components.len]u8 = undefined;
    comptime var i: usize = 0;
    inline for (components) |c| {
        switch (c) {
            'x' => mask[i] = 0,
            'y' => mask[i] = 1,
            'z' => mask[i] = 2,
            'w' => mask[i] = 3,
            else => @compileError("swizzle: invalid component"),
        }
        i += 1;
    }

    return @shuffle(
        T,
        v,
        @as(@Vector(1, T), undefined),
        mask,
    );
}

pub fn norm_sqr_adv(vec: anytype, comptime precision: u8) Float(precision) {
    const T = meta.Child(@TypeOf(vec));
    // Array inputs are processed in native-width SIMD chunks (see simd.zig) so a
    // large array never materializes one huge @Vector.
    if (@typeInfo(@TypeOf(vec)) == .array) {
        if (@typeInfo(T) == .int) return @floatFromInt(simd.sumSquares(T, vec));
        const C = if (precision > @bitSizeOf(T)) Float(precision) else T;
        return @floatCast(simd.sumSquares(C, vec));
    }
    const v: meta.AsVector(@TypeOf(vec)) = vec;
    if (@typeInfo(T) == .int) {
        return @as(
            Float(precision),
            @floatFromInt(@reduce(
                .Add,
                v * v,
            )),
        );
    } else {
        const items: blk: {
            if (precision > @bitSizeOf(T)) {
                break :blk @Vector(meta.lengthOf(@TypeOf(vec)), Float(precision));
            } else {
                break :blk meta.AsVector(@TypeOf(vec));
            }
        } = v;

        return @floatCast(@reduce(
            .Add,
            items * items,
        ));
    }
}

pub inline fn norm_sqr(vec: anytype) Float(@bitSizeOf(meta.Child(@TypeOf(vec)))) {
    return norm_sqr_adv(vec, @bitSizeOf(meta.Child(@TypeOf(vec))));
}

/// Returns the norm of the vector as a Float.
///
/// the precsion parameter is the number of bits of the output.
/// the precision of the calculations will match the precision of the output type.
pub fn norm_adv(vec: anytype, comptime precision: u8) Float(precision) {
    const T = meta.Child(@TypeOf(vec));
    if (@typeInfo(@TypeOf(vec)) == .array) {
        if (@typeInfo(T) == .int)
            return std.math.sqrt(@as(Float(precision), @floatFromInt(simd.sumSquares(T, vec))));
        const C = if (precision > @bitSizeOf(T)) Float(precision) else T;
        return @floatCast(std.math.sqrt(simd.sumSquares(C, vec)));
    }
    const v: meta.AsVector(@TypeOf(vec)) = vec;
    if (@typeInfo(T) == .int) {
        return std.math.sqrt(
            @as(
                Float(precision),
                @floatFromInt(@reduce(
                    .Add,
                    v * v,
                )),
            ),
        );
    } else {
        const items: blk: {
            if (precision > @bitSizeOf(T)) {
                break :blk @Vector(meta.lengthOf(@TypeOf(vec)), Float(precision));
            } else {
                break :blk meta.AsVector(@TypeOf(vec));
            }
        } = v;

        return @floatCast(std.math.sqrt(
            @reduce(
                .Add,
                items * items,
            ),
        ));
    }
}

/// Returns the norm of the vector as a Float with the default precision.
/// the precision of the output is the number of bits of the output.
/// see `normAdv` for more information.
pub inline fn norm(vec: anytype) Float(@bitSizeOf(meta.Child(@TypeOf(vec)))) {
    return norm_adv(vec, @bitSizeOf(meta.Child(@TypeOf(vec))));
}

/// Returns a new vector with the same direction as the original vector, but with a norm closest to 1.
pub fn normalize(vec: anytype) @TypeOf(vec) {
    const T = meta.Child(@TypeOf(vec));
    if (@typeInfo(@TypeOf(vec)) == .array) {
        const self_norm = switch (@typeInfo(T)) {
            .float => norm(vec),
            .int => @as(T, @intFromFloat(norm(vec))),
            else => unreachable,
        };
        return simd.mapUnary(vec, self_norm, struct {
            fn op(c: anytype, d: T) @TypeOf(c) {
                return c / @as(@TypeOf(c), @splat(d));
            }
        }.op);
    }
    const v: meta.AsVector(@TypeOf(vec)) = vec;
    const self_norm = switch (@typeInfo(T)) {
        .float => norm(v),
        .int => @as(T, @intFromFloat(norm(v))),
        else => unreachable,
    };
    return v / @as(
        meta.AsVector(@TypeOf(vec)),
        @splat(self_norm),
    );
}

/// Like `normalize`, but returns `fallback` when the vector is too short to have a
/// meaningful direction. `normalize` divides by the norm unconditionally, so a zero
/// vector yields NaN; this is the guarded form for inputs that may be degenerate.
///
/// The threshold is on the *squared* norm, so no square root is taken on the reject path.
/// Container kind is preserved: a `@Vector` in yields a `@Vector`, an array yields an array.
pub fn normalize_safe(
    vec: anytype,
    fallback: @TypeOf(vec),
    min_norm_sqr: Float(@bitSizeOf(meta.Child(@TypeOf(vec)))),
) @TypeOf(vec) {
    if (norm_sqr(vec) <= min_norm_sqr) return fallback;
    return normalize(vec);
}

/// `normalize_safe` with a default threshold of 1.0e-6 on the squared norm.
pub fn normalize_safe_default(vec: anytype, fallback: @TypeOf(vec)) @TypeOf(vec) {
    return normalize_safe(vec, fallback, 1.0e-6);
}

/// dot product of two vectors.
pub fn dot(vec: anytype, other: @TypeOf(vec)) meta.Child(@TypeOf(vec)) {
    if (@typeInfo(@TypeOf(vec)) == .array)
        return simd.sumProducts(meta.Child(@TypeOf(vec)), vec, other);
    const a: meta.AsVector(@TypeOf(vec)) = vec;
    const b: meta.AsVector(@TypeOf(vec)) = other;
    return @reduce(.Add, a * b);
}

pub fn norm_perpendicular(a: anytype) @TypeOf(a) {
    comptime {
        std.debug.assert(meta.lengthOf(@TypeOf(a)) == 3);
    }
    const V = meta.AsVector(@TypeOf(a));
    const v: V = a;
    if (@abs(v[0]) > @abs(v[1])) {
        const len = norm(swizzle(v, "xz"));
        return V{ v[2], 0.0, -v[0] } / @as(V, @splat(len));
    } else {
        const len = norm(swizzle(v, "yz"));
        return V{ 0.0, v[2], -v[1] } / @as(V, @splat(len));
    }
}

/// Returns the cross product of two vectors.
pub fn cross(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    comptime {
        std.debug.assert(meta.lengthOf(@TypeOf(a)) == 3);
    }
    const T = meta.Child(@TypeOf(a));
    const av: meta.AsVector(@TypeOf(a)) = a;
    const bv: meta.AsVector(@TypeOf(a)) = b;
    const self1 = @shuffle(T, av, av, [3]u8{ 1, 2, 0 });
    const self2 = @shuffle(T, av, av, [3]u8{ 2, 0, 1 });
    const other1 = @shuffle(T, bv, bv, [3]u8{ 1, 2, 0 });
    const other2 = @shuffle(T, bv, bv, [3]u8{ 2, 0, 1 });

    return self1 * other2 - self2 * other1;
}

/// Returns the distance between two vectors.
///
/// the precsion parameter is the number of bits of the Vector type T.
/// the precision of the calculations will match the precision of the output type.
pub fn distance_adv(a: anytype, b: @TypeOf(a), comptime precision: u8) Float(precision) {
    if (@typeInfo(@TypeOf(a)) == .array)
        return norm_adv(simd.sub(a, b), precision); // sub -> [N]T, norm_adv chunks it
    const av: meta.AsVector(@TypeOf(a)) = a;
    const bv: meta.AsVector(@TypeOf(a)) = b;
    return norm_adv(av - bv, precision);
}

/// Returns the distance between two vectors.
///
/// the precision of the output is the number of bits of T.
/// see `distanceAdv` for more information.
pub inline fn distance(vec: anytype, other: @TypeOf(vec)) Float(@bitSizeOf(meta.Child(@TypeOf(vec)))) {
    return distance_adv(vec, other, @bitSizeOf(meta.Child(@TypeOf(vec))));
}

/// Returns the angle between two vectors.
///
/// the precsion parameter is the number of bits of the Vector type T.
/// the precision of the calculations will match the precision of the output type.
pub fn angle_adv(vec: anytype, other: @TypeOf(vec), comptime precision: u8) Float(precision) {
    return std.math.acos(dot(vec, other) / (norm_adv(vec, precision) * norm_adv(other, precision)));
}

/// Returns the angle between two vectors.
pub inline fn angle(a: anytype, b: @TypeOf(a)) meta.Child(@TypeOf(a)) {
    return angle_adv(a, b, @bitSizeOf(meta.Child(@TypeOf(a))));
}

/// Returns a new vector that is the reflection of the original vector on the given normal.
pub fn reflect(vec: anytype, normal: @TypeOf(vec)) @TypeOf(vec) {
    const T = meta.Child(@TypeOf(vec));
    if (@typeInfo(@TypeOf(vec)) == .array) {
        const dp = dot(vec, normal); // chunked dot
        return simd.mapBinary(vec, normal, dp, struct {
            fn op(cv: anytype, cn: anytype, d: T) @TypeOf(cv) {
                const two: @TypeOf(cv) = @splat(2);
                const dd: @TypeOf(cv) = @splat(d);
                return cv - cn * two * dd;
            }
        }.op);
    }
    const V = meta.AsVector(@TypeOf(vec));
    const v: V = vec;
    const n: V = normal;
    const dot_product = dot(v, n);
    return v - (n *
        @as(V, @splat(2)) *
        @as(V, @splat(dot_product)));
}

/// Simultaneously computes the sine and cosine of each component.
/// The cephes-derived core lives in `simd.sinCosVec`; array inputs are processed
/// in native-width SIMD chunks via `simd.sinCos`, `@Vector` inputs in one op.
pub fn sin_cos(input: anytype) struct { sin_out: @TypeOf(input), cos_out: @TypeOf(input) } {
    if (@typeInfo(@TypeOf(input)) == .array) {
        const r = simd.sinCos(input);
        return .{ .sin_out = r.sin, .cos_out = r.cos };
    }
    const r = simd.sinCosVec(input); // input is a @Vector -> width == its len
    return .{ .sin_out = r.sin, .cos_out = r.cos };
}

/// Returns a new vector with a direction closest to the original vector, but with a magnitude scaled by the given value.
pub inline fn scale(a: anytype, value: meta.Child(@TypeOf(a))) @TypeOf(a) {
    if (@typeInfo(@TypeOf(a)) == .array)
        return simd.mapUnary(a, value, struct {
            fn op(c: anytype, s: meta.Child(@TypeOf(a))) @TypeOf(c) {
                return c * @as(@TypeOf(c), @splat(s));
            }
        }.op);
    const V = meta.AsVector(@TypeOf(a));
    const v: V = a;
    return v * @as(V, @splat(value));
}

/// Element-wise `a + b`, preserving the container kind (array in -> array out).
pub fn add(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    if (@typeInfo(@TypeOf(a)) == .array)
        return simd.mapBinary(a, b, {}, struct {
            fn op(x: anytype, y: anytype, _: void) @TypeOf(x) {
                return x + y;
            }
        }.op);
    const av: meta.AsVector(@TypeOf(a)) = a;
    const bv: meta.AsVector(@TypeOf(a)) = b;
    return av + bv;
}

/// Element-wise `a - b`, preserving the container kind (array in -> array out).
pub fn sub(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    if (@typeInfo(@TypeOf(a)) == .array)
        return simd.sub(a, b);
    const av: meta.AsVector(@TypeOf(a)) = a;
    const bv: meta.AsVector(@TypeOf(a)) = b;
    return av - bv;
}

/// Element-wise (Hadamard) `a * b`, preserving the container kind.
pub fn mul(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    if (@typeInfo(@TypeOf(a)) == .array)
        return simd.mapBinary(a, b, {}, struct {
            fn op(x: anytype, y: anytype, _: void) @TypeOf(x) {
                return x * y;
            }
        }.op);
    const av: meta.AsVector(@TypeOf(a)) = a;
    const bv: meta.AsVector(@TypeOf(a)) = b;
    return av * bv;
}

/// Element-wise `a / b`, preserving the container kind.
pub fn div(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    if (@typeInfo(@TypeOf(a)) == .array)
        return simd.mapBinary(a, b, {}, struct {
            fn op(x: anytype, y: anytype, _: void) @TypeOf(x) {
                return x / y;
            }
        }.op);
    const av: meta.AsVector(@TypeOf(a)) = a;
    const bv: meta.AsVector(@TypeOf(a)) = b;
    return av / bv;
}

/// Squared distance between two vectors. Cheaper than `distance` when only comparisons are needed.
pub inline fn distance_sqr(a: anytype, b: @TypeOf(a)) Float(@bitSizeOf(meta.Child(@TypeOf(a)))) {
    if (@typeInfo(@TypeOf(a)) == .array)
        return norm_sqr(simd.sub(a, b));
    const av: meta.AsVector(@TypeOf(a)) = a;
    const bv: meta.AsVector(@TypeOf(a)) = b;
    return norm_sqr(av - bv);
}

/// Exact (bitwise) equality of two vectors, container-kind agnostic.
pub fn eql(a: anytype, b: @TypeOf(a)) bool {
    const av: meta.AsVector(@TypeOf(a)) = a;
    const bv: meta.AsVector(@TypeOf(a)) = b;
    return @reduce(.And, av == bv);
}

pub fn is_close_default(a: anytype, b: @TypeOf(a)) bool {
    return is_close(a, b, 1.0e-12);
}

pub fn is_close(a: anytype, b: @TypeOf(a), max_distance_sqr: Float(@bitSizeOf(meta.Child(@TypeOf(a))))) bool {
    if (@typeInfo(@TypeOf(a)) == .array)
        return norm_sqr(simd.sub(a, b)) <= max_distance_sqr;
    const av: meta.AsVector(@TypeOf(a)) = a;
    const bv: meta.AsVector(@TypeOf(a)) = b;
    return norm_sqr(av - bv) <= max_distance_sqr;
}

pub fn is_normalized_default(a: anytype) bool {
    return is_normalized(a, 1.0e-6);
}

pub fn is_normalized(a: anytype, tolerance: Float(@bitSizeOf(meta.Child(@TypeOf(a))))) bool {
    return @abs(norm_sqr(a) - 1.0) <= tolerance;
}

/// Encode a unit vector into a 2-component octahedral representation in [-1, 1]^2.
/// Useful for compact normal storage. See `decode_oct` for the inverse.
pub fn encode_oct(n: anytype) @Vector(2, meta.Child(@TypeOf(n))) {
    const T = meta.Child(@TypeOf(n));
    const l1 = @abs(n[0]) + @abs(n[1]) + @abs(n[2]);
    var p = @Vector(2, T){ n[0] / l1, n[1] / l1 };
    if (n[2] <= 0) {
        // Capture components first: `p = .{...}` builds in place, so reading
        // p[0]/p[1] on the RHS would otherwise see partially-updated values.
        const px = p[0];
        const py = p[1];
        const sx: T = if (px >= 0) 1 else -1;
        const sy: T = if (py >= 0) 1 else -1;
        p = .{ (1 - @abs(py)) * sx, (1 - @abs(px)) * sy };
    }
    return p;
}

/// Decode a 2-component octahedral representation back into a unit vector.
pub fn decode_oct(e: anytype) @Vector(3, meta.Child(@TypeOf(e))) {
    const T = meta.Child(@TypeOf(e));
    var n = @Vector(3, T){ e[0], e[1], 1 - @abs(e[0]) - @abs(e[1]) };
    const t = @max(-n[2], 0);
    n[0] += if (n[0] >= 0) -t else t;
    n[1] += if (n[1] >= 0) -t else t;
    return normalize(n);
}

/// Build a right-handed orthonormal basis (tangent, bitangent) around a unit
/// normal using Duff et al.'s branchless method. `n` is the third basis vector.
pub fn build_basis(n: anytype) struct { tangent: @TypeOf(n), bitangent: @TypeOf(n) } {
    const T = meta.Child(@TypeOf(n));
    const s: T = std.math.copysign(@as(T, 1), n[2]);
    const a = -1.0 / (s + n[2]);
    const b = n[0] * n[1] * a;
    return .{
        .tangent = .{ 1.0 + s * n[0] * n[0] * a, s * b, -s * n[0] },
        .bitangent = .{ b, s + n[1] * n[1] * a, -n[1] },
    };
}

test encode_oct {
    const dirs = [_]@Vector(3, f32){
        .{ 0, 0, 1 },
        .{ 0, 0, -1 },
        .{ 1, 0, 0 },
        .{ 0, -1, 0 },
        .{ 1, 2, 3 },
        .{ -3, 2, -1 },
        .{ 0.5, -0.5, -0.7 },
        .{ -2, -2, -2 },
    };
    for (dirs) |d| {
        const dn = normalize(d);
        try std.testing.expect(is_close(decode_oct(encode_oct(dn)), dn, 1e-6));
    }
}

test build_basis {
    const n = normalize(@Vector(3, f32){ 0.3, -0.5, 0.8 });
    const basis = build_basis(n);
    try std.testing.expectApproxEqAbs(@as(f32, 0), dot(basis.tangent, n), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), dot(basis.bitangent, n), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), dot(basis.tangent, basis.bitangent), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1), norm(basis.tangent), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1), norm(basis.bitangent), 1e-5);
}

test scale {
    const v = @Vector(2, f32){ 1, 2 };
    try std.testing.expectEqual(@Vector(2, f32){ 2, 4 }, scale(v, 2));
    try std.testing.expectEqual(@Vector(2, f32){ 0.5, 1 }, scale(v, 0.5));
}

test "add sub mul div distance_sqr eql" {
    const a = @Vector(3, f32){ 1, 2, 3 };
    const b = @Vector(3, f32){ 4, 5, 6 };
    try std.testing.expectEqual(@Vector(3, f32){ 5, 7, 9 }, add(a, b));
    try std.testing.expectEqual(@Vector(3, f32){ -3, -3, -3 }, sub(a, b));
    try std.testing.expectEqual(@Vector(3, f32){ 4, 10, 18 }, mul(a, b));
    try std.testing.expectEqual(@Vector(3, f32){ 4, 2.5, 2 }, div(b, a));
    try std.testing.expectEqual(@as(f32, 27), distance_sqr(a, b));
    try std.testing.expect(eql(a, a));
    try std.testing.expect(!eql(a, b));

    // Array inputs preserve the array container.
    const aa = [3]f32{ 1, 2, 3 };
    const ba = [3]f32{ 4, 5, 6 };
    try std.testing.expect(@TypeOf(add(aa, ba)) == [3]f32);
    try std.testing.expectEqual([3]f32{ 5, 7, 9 }, add(aa, ba));
    try std.testing.expectEqual([3]f32{ 4, 10, 18 }, mul(aa, ba));
    try std.testing.expectEqual(@as(f32, 27), distance_sqr(aa, ba));
    try std.testing.expect(eql(aa, aa));
    try std.testing.expect(!eql(aa, ba));
}

test is_close {
    try std.testing.expect(is_close(@Vector(4, f32){ 1, 2, 3, 4 }, @Vector(4, f32){ 1.001, 2.001, 3.001, 4.001 }, 1.0e-4));
    try std.testing.expect(!is_close(@Vector(4, f32){ 1, 2, 3, 4 }, @Vector(4, f32){ 1.001, 2.001, 3.001, 4.001 }, 1.0e-6));
}

test swizzle {
    const v = @Vector(2, f32){ 1, 2 };
    try std.testing.expectEqual(@as(f32, 2), swizzle(v, "yx")[0]);
    try std.testing.expectEqual(@as(f32, 1), swizzle(v, "yx")[1]);
    const v2 = @Vector(3, f32){ 1, 2, 3 };
    const v2_expected = @Vector(3, f32){ 2, 3, 1 };
    try std.testing.expectEqual(v2_expected, swizzle(v2, "yzx"));
}

test angle_adv {
    const v1 = @Vector(2, f32){ 1, 0 };
    const v2 = @Vector(2, f32){ 0, 1 };
    const expected: f64 = @as(f64, std.math.pi) / @as(f64, 2);
    try std.testing.expectApproxEqAbs(expected, angle_adv(v1, v2, 32), 0.0000001);
    try std.testing.expectApproxEqAbs(expected, angle_adv(v1, v2, 64), 0.00000000001);
}

test normalize {
    // Test with floating point vector
    const normalized_f32 = normalize(@Vector(3, f32){ 3, 4, 0 });
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), normalized_f32[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), normalized_f32[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0), normalized_f32[2], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 1), norm(normalized_f32), 0.0001);
    // Test with integer vector
    const v_i32 = @Vector(2, i32){ 3, 4 };
    const normalized_i32 = normalize(v_i32);
    try std.testing.expectEqual(@Vector(2, i32){ 0, 0 }, normalized_i32); // Integer division limitations
}

test normalize_safe {
    const fallback = @Vector(3, f32){ 1, 0, 0 };

    // Degenerate input takes the fallback instead of dividing through to NaN.
    try std.testing.expectEqual(fallback, normalize_safe_default(@Vector(3, f32){ 0, 0, 0 }, fallback));
    try std.testing.expect(std.math.isNan(normalize(@Vector(3, f32){ 0, 0, 0 })[0]));

    // Non-degenerate input agrees with `normalize`.
    const v = @Vector(3, f32){ 3, 4, 0 };
    try std.testing.expectEqual(normalize(v), normalize_safe_default(v, fallback));

    // The threshold is on the squared norm, so a vector shorter than it is rejected.
    const tiny = @Vector(3, f32){ 1.0e-4, 0, 0 }; // norm_sqr == 1.0e-8
    try std.testing.expectEqual(fallback, normalize_safe_default(tiny, fallback));
    try std.testing.expectEqual(normalize(tiny), normalize_safe(tiny, fallback, 1.0e-12));

    // Container kind is preserved: an array in yields an array out.
    const array_fallback = [3]f32{ 1, 0, 0 };
    try std.testing.expectEqual(array_fallback, normalize_safe_default([3]f32{ 0, 0, 0 }, array_fallback));
    const normalized_array = normalize_safe_default([3]f32{ 3, 4, 0 }, array_fallback);
    try std.testing.expectEqual([3]f32, @TypeOf(normalized_array));
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), normalized_array[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), normalized_array[1], 0.0001);
}

test dot {
    const v1 = @Vector(3, f32){ 1, 2, 3 };
    const v2 = @Vector(3, f32){ 4, 5, 6 };
    try std.testing.expectEqual(@as(f32, 32), dot(v1, v2));

    const v3 = @Vector(2, i32){ 2, 3 };
    const v4 = @Vector(2, i32){ 4, 5 };
    try std.testing.expectEqual(@as(i32, 23), dot(v3, v4));
}

test cross {
    const v1 = @Vector(3, f32){ 1, 0, 0 };
    const v2 = @Vector(3, f32){ 0, 1, 0 };
    const expected = @Vector(3, f32){ 0, 0, 1 };
    try std.testing.expectEqual(expected, cross(v1, v2));

    const v3 = @Vector(3, i32){ 2, 3, 4 };
    const v4 = @Vector(3, i32){ 5, 6, 7 };
    const expected_i = @Vector(3, i32){ -3, 6, -3 };
    try std.testing.expectEqual(expected_i, cross(v3, v4));
}

test distance {
    const v1 = @Vector(2, f32){ 1, 1 };
    const v2 = @Vector(2, f32){ 4, 5 };
    try std.testing.expectApproxEqAbs(@as(f32, 5), distance(v1, v2), 0.0001);

    const v3 = @Vector(3, f64){ 1, 2, 3 };
    const v4 = @Vector(3, f64){ 4, 6, 8 };
    try std.testing.expectApproxEqAbs(@as(f64, 7.0710678118654755), distance(v3, v4), 0.0000001);
}

test angle {
    const v1 = @Vector(2, f32){ 1, 0 };
    const v2 = @Vector(2, f32){ 0, 1 };
    try std.testing.expectApproxEqAbs(@as(f32, std.math.pi / 2.0), angle(v1, v2), 0.0001);

    const v3 = @Vector(2, f32){ 1, 1 };
    const v4 = @Vector(2, f32){ -1, 1 };
    try std.testing.expectApproxEqAbs(@as(f32, std.math.pi / 2.0), angle(v3, v4), 0.0001);
}

test reflect {
    const v = @Vector(2, f32){ 1, -1 };
    const normal = @Vector(2, f32){ 0, 1 };
    const expected = @Vector(2, f32){ 1, 1 };
    try std.testing.expectEqual(expected, reflect(v, normal));

    const v2 = @Vector(3, f32){ 1, -1, 0.5 };
    const normal2 = @Vector(3, f32){ 0, 1, 0 };
    const expected2 = @Vector(3, f32){ 1, 1, 0.5 };

    const result = reflect(v2, normal2);
    try std.testing.expectApproxEqAbs(expected2[0], result[0], 0.0001);
    try std.testing.expectApproxEqAbs(expected2[1], result[1], 0.0001);
    try std.testing.expectApproxEqAbs(expected2[2], result[2], 0.0001);
}

test norm_adv {
    const v = @Vector(2, f64){ 1, 0 };
    try std.testing.expectEqual(@as(f32, 1), norm_adv(v, 32));
    const v2 = @Vector(2, f16){ 1, 2 };
    const result = norm_adv(v2, 64);
    try std.testing.expect(@TypeOf(result) == f64);
    const expected = 2.2360679774997896964091736687;
    try std.testing.expectEqual(@as(f16, expected), norm_adv(v2, 16));
    try std.testing.expectEqual(@as(f32, expected), norm_adv(v2, 32));
    try std.testing.expectEqual(@as(f64, expected), norm_adv(v2, 64));
    const v3 = @Vector(2, i8){ 1, 2 };
    try std.testing.expect(@TypeOf(norm_adv(v3, 64)) == f64);
    try std.testing.expectEqual(@as(f16, expected), norm_adv(v3, 16));
    try std.testing.expectEqual(@as(f32, expected), norm_adv(v3, 32));
    try std.testing.expectEqual(@as(f64, expected), norm_adv(v3, 64));
    const v4 = @Vector(2, u8){ 1, 2 };
    try std.testing.expect(@TypeOf(norm_adv(v4, 64)) == f64);
    try std.testing.expectEqual(@as(f16, expected), norm_adv(v4, 16));
    try std.testing.expectEqual(@as(f32, expected), norm_adv(v4, 32));
    try std.testing.expectEqual(@as(f64, expected), norm_adv(v4, 64));
}

test norm {
    const v1 = @Vector(2, f32){ 3, 4 };
    try std.testing.expectEqual(@as(f32, 5), norm(v1));

    const v2 = @Vector(3, f64){ 1, 2, 2 };
    try std.testing.expectApproxEqAbs(@as(f64, 3), norm(v2), 0.0001);

    const v3 = @Vector(4, i32){ 1, 2, 2, 0 };
    try std.testing.expectApproxEqAbs(@as(f32, 3), norm(v3), 0.0001);
}

test sin_cos {
    {
        const input = @Vector(2, f32){ 0, std.math.pi * 0.5 };
        const res = sin_cos(input);
        try testing.expect_is_close(res.cos_out, @Vector(2, f32){ 1, 0 }, 0.0001);
        try testing.expect_is_close(res.sin_out, @Vector(2, f32){ 0, 1.0 }, 0.0001);
    }
    {
        const input = @Vector(4, f32){ 0, std.math.pi * 0.5, std.math.pi, -0.5 * std.math.pi };
        const res = sin_cos(input);
        try testing.expect_is_close(res.cos_out, @Vector(4, f32){ 1, 0, -1, 0 }, 0.0001);
        try testing.expect_is_close(res.sin_out, @Vector(4, f32){ 0, 1.0, 0, -1.0 }, 0.0001);
    }
    var ms: f64 = 0.0;
    var mc: f64 = 0.0;

    var x: f32 = -100.0 * std.math.pi;
    while (x < 100.0 * std.math.pi) : (x += 1.0e-3) {
        const xv = @as(Vec4f32, @splat(x)) + Vec4f32{ 0.0e-4, 2.5e-4, 5.0e-4, 7.5e-4 };
        const res2 = sin_cos(xv);
        inline for (0..3) |i| {
            const s1 = std.math.sin(xv[i]);
            const s2 = res2.sin_out[i];
            ms = @max(ms, @abs(s1 - s2));

            const c1 = std.math.cos(xv[i]);
            const c2 = res2.cos_out[i];
            mc = @max(mc, @abs(c1 - c2));
        }
    }
    try std.testing.expect(ms < 1.0e-7);
    try std.testing.expect(mc < 1.0e-7);
}

test "array inputs are accepted and preserve shape" {
    // Same-length vector-returning: array in -> array out.
    {
        const result = normalize([3]f32{ 3, 4, 0 });
        try std.testing.expect(@TypeOf(result) == [3]f32);
        try std.testing.expectApproxEqAbs(@as(f32, 0.6), result[0], 0.0001);
        try std.testing.expectApproxEqAbs(@as(f32, 0.8), result[1], 0.0001);
        try std.testing.expectApproxEqAbs(@as(f32, 0), result[2], 0.0001);
    }
    // @Vector in -> @Vector out (unchanged behavior).
    {
        const result = normalize(@Vector(3, f32){ 3, 4, 0 });
        try std.testing.expect(@TypeOf(result) == @Vector(3, f32));
    }
    // Scalar-returning: container-independent.
    {
        const d = dot([3]f32{ 1, 2, 3 }, [3]f32{ 4, 5, 6 });
        try std.testing.expect(@TypeOf(d) == f32);
        try std.testing.expectEqual(@as(f32, 32), d);
        try std.testing.expectApproxEqAbs(@as(f32, 5), norm([2]f32{ 3, 4 }), 0.0001);
    }
    // cross: array in -> array out.
    {
        const result = cross([3]f32{ 1, 0, 0 }, [3]f32{ 0, 1, 0 });
        try std.testing.expect(@TypeOf(result) == [3]f32);
        try std.testing.expectApproxEqAbs(@as(f32, 1), result[2], 0.0001);
    }
    // reflect: array in -> array out.
    {
        const result = reflect([2]f32{ 1, -1 }, [2]f32{ 0, 1 });
        try std.testing.expect(@TypeOf(result) == [2]f32);
        try std.testing.expectApproxEqAbs(@as(f32, 1), result[1], 0.0001);
    }
    // scale: array in -> array out.
    {
        const result = scale([2]f32{ 1, 2 }, 2);
        try std.testing.expect(@TypeOf(result) == [2]f32);
        try std.testing.expectApproxEqAbs(@as(f32, 4), result[1], 0.0001);
    }
    // Length-changing: extract/swizzle preserve the container kind.
    {
        const e = extract([4]f32{ 1, 2, 3, 4 }, 2);
        try std.testing.expect(@TypeOf(e) == [2]f32);
        try std.testing.expectEqual(@as(f32, 1), e[0]);
        try std.testing.expectEqual(@as(f32, 2), e[1]);

        const s = swizzle([2]f32{ 1, 2 }, "yx");
        try std.testing.expect(@TypeOf(s) == [2]f32);
        try std.testing.expectEqual(@as(f32, 2), s[0]);
        try std.testing.expectEqual(@as(f32, 1), s[1]);
    }
    // distance / angle over arrays.
    {
        try std.testing.expectApproxEqAbs(@as(f32, 5), distance([2]f32{ 1, 1 }, [2]f32{ 4, 5 }), 0.0001);
        try std.testing.expectApproxEqAbs(@as(f32, std.math.pi / 2.0), angle([2]f32{ 1, 0 }, [2]f32{ 0, 1 }), 0.0001);
    }
    // is_close over arrays.
    {
        try std.testing.expect(is_close([2]f32{ 1, 2 }, [2]f32{ 1.0001, 2.0001 }, 1.0e-4));
    }
    // sin_cos: array in -> array out.
    {
        const res = sin_cos([2]f32{ 0, std.math.pi * 0.5 });
        try std.testing.expect(@TypeOf(res.sin_out) == [2]f32);
        try testing.expect_is_close(@as(@Vector(2, f32), res.cos_out), @Vector(2, f32){ 1, 0 }, 0.0001);
        try testing.expect_is_close(@as(@Vector(2, f32), res.sin_out), @Vector(2, f32){ 0, 1.0 }, 0.0001);
    }
}

test "chunked array path matches single-@Vector path" {
    @setEvalBranchQuota(10000);
    // For each length (incl. remainder-forcing and the motivating 245), the
    // chunked array kernels must agree with the single-op @Vector path on the
    // same data. Pure element-wise ops (scale, sin_cos) must match exactly;
    // reductions and reduction-dependent maps agree to within float reordering.
    // Inner fill/compare loops are RUNTIME `for` (not `inline`) so the unrolled
    // outer length loop doesn't blow the comptime branch quota.
    const lengths = [_]comptime_int{ 3, 8, 16, 17, 64, 65, 77, 245 };
    inline for (lengths) |N| {
        // Strictly-positive, moderate f32 data so reduction magnitudes stay nonzero.
        var a: [N]f32 = undefined;
        var b: [N]f32 = undefined;
        for (0..N) |i| {
            a[i] = @as(f32, @floatFromInt((i % 13) + 1)) * 0.5; // 0.5 .. 6.5
            b[i] = @as(f32, @floatFromInt((i % 7) + 1)) * 0.25; // 0.25 .. 1.75
        }
        const va: @Vector(N, f32) = a;
        const vb: @Vector(N, f32) = b;

        // Reductions -> scalar: relative tolerance absorbs the sum reordering.
        try std.testing.expectApproxEqRel(norm_sqr(va), norm_sqr(a), 1e-4);
        try std.testing.expectApproxEqRel(norm(va), norm(a), 1e-4);
        try std.testing.expectApproxEqRel(dot(va, vb), dot(a, b), 1e-4);
        try std.testing.expectApproxEqRel(distance(va, vb), distance(a, b), 1e-4);

        // scale: pure per-lane multiply -> bit-identical.
        try std.testing.expectEqual(@as([N]f32, scale(va, 2.0)), scale(a, 2.0));

        // sin_cos: pure per-lane transcendental -> bit-identical.
        const sc_v = sin_cos(va);
        const sc_a = sin_cos(a);
        try std.testing.expectEqual(@as([N]f32, sc_v.sin_out), sc_a.sin_out);
        try std.testing.expectEqual(@as([N]f32, sc_v.cos_out), sc_a.cos_out);

        // normalize: components are O(1/sqrt(N)); only the norm scalar reorders.
        const nz_v: [N]f32 = normalize(va);
        const nz_a = normalize(a);
        for (0..N) |i| try std.testing.expectApproxEqAbs(nz_v[i], nz_a[i], 1e-4);

        // reflect about a UNIT normal keeps magnitudes O(input) so an abs tol works.
        const unit_v: [N]f32 = normalize(vb);
        const unit_a = normalize(b);
        const rf_v: [N]f32 = reflect(va, @as(@Vector(N, f32), unit_v));
        const rf_a = reflect(a, unit_a);
        for (0..N) |i| try std.testing.expectApproxEqAbs(rf_v[i], rf_a[i], 1e-3);

        // is_close over arrays agrees with the vector path.
        try std.testing.expectEqual(is_close(va, vb, 1000.0), is_close(a, b, 1000.0));
    }

    // Integer reductions/maps are exact (integer add is associative mod 2^bits).
    inline for (.{ 3, 17, 64, 245 }) |N| {
        var ai: [N]i32 = undefined;
        var bi: [N]i32 = undefined;
        for (0..N) |i| {
            ai[i] = @intCast((i % 5) + 1);
            bi[i] = @intCast((i % 3) + 1);
        }
        const vai: @Vector(N, i32) = ai;
        const vbi: @Vector(N, i32) = bi;
        try std.testing.expectEqual(norm_sqr(vai), norm_sqr(ai));
        try std.testing.expectEqual(dot(vai, vbi), dot(ai, bi));
        try std.testing.expectEqual(@as([N]i32, normalize(vai)), normalize(ai));
    }

    // f16 with precision-64 widening: both paths accumulate in f64, so they match.
    inline for (.{ 17, 64 }) |N| {
        var af: [N]f16 = undefined;
        for (0..N) |i| af[i] = @floatFromInt((i % 4) + 1);
        const vaf: @Vector(N, f16) = af;
        try std.testing.expectApproxEqRel(norm_adv(vaf, 64), norm_adv(af, 64), 1e-6);
    }

    // f64 reductions.
    inline for (.{ 64, 245 }) |N| {
        var ad: [N]f64 = undefined;
        for (0..N) |i| ad[i] = @as(f64, @floatFromInt((i % 11) + 1)) * 0.3;
        const vad: @Vector(N, f64) = ad;
        try std.testing.expectApproxEqRel(norm(vad), norm(ad), 1e-12);
    }
}
