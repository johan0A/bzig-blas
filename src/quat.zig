const std = @import("std");
const vector = @import("vector.zig");
const meta = @import("meta.zig");
const zml = @import("root.zig");

pub const Quat4f32 = @Vector(4, f32);
pub const Quat4f64 = @Vector(4, f64);

fn map_to_vector(a: anytype) meta.AsVector(@TypeOf(a)) {
    return a;
}

fn Quat(comptime T: type) type {
    return @Vector(4, T);
}

pub fn identity(comptime T: type) @Vector(4, T) {
    return .{ 0, 0, 0, 1 };
}

pub fn zero(comptime T: type) @Vector(4, T) {
    return .{ 0, 0, 0, 0 };
}

pub fn x_axis(comptime T: type) @Vector(3, T) {
    return .{ 1, 0, 0 };
}

pub fn y_axis(comptime T: type) @Vector(3, T) {
    return .{ 0, 1, 0 };
}

pub fn z_axis(comptime T: type) @Vector(3, T) {
    return .{ 0, 0, 1 };
}

// Create quaternion that rotates a vector from the direction of inFrom to the direction of inTo along the shortest path
// @see https://www.euclideanspace.com/maths/algebra/vectors/angleBetween/index.htm
pub fn from_to(from: anytype, to: @TypeOf(from)) @Vector(4, std.meta.Child(@TypeOf(from))) {
    comptime {
        std.debug.assert(@typeInfo(@TypeOf(from)) == .vector);
        std.debug.assert(@typeInfo(@TypeOf(from)).vector.len == 3);
    }

    //Uses (inFrom = v1, inTo = v2):

    //angle = arcos(v1 . v2 / |v1||v2|)
    //axis = normalize(v1 x v2)

    //Quaternion is then:

    //s = sin(angle / 2)
    //x = axis.x * s
    //y = axis.y * s
    //z = axis.z * s
    //w = cos(angle / 2)

    //Using identities:

    //sin(2 * a) = 2 * sin(a) * cos(a)
    //cos(2 * a) = cos(a)^2 - sin(a)^2
    //sin(a)^2 + cos(a)^2 = 1

    //This reduces to:

    //x = (v1 x v2).x
    //y = (v1 x v2).y
    //z = (v1 x v2).z
    //w = |v1||v2| + v1 . v2

    //which then needs to be normalized because the whole equation was multiplied by 2 cos(angle / 2)
    const len_v1_v2 = std.math.sqrt(vector.norm_sqr(from) * vector.norm_sqr(to));
    const w = len_v1_v2 + vector.dot(from, to);
    if (w == 0.0) {
        if (len_v1_v2 == 0.0) {
            return identity(std.meta.Child(@TypeOf(from)));
        } else {
            const norm_perp = vector.norm_perpendicular(from);
            return .{ norm_perp[0], norm_perp[1], norm_perp[2], 0 };
        }
    }
    const v = vector.cross(from, to);
    return vector.normalize(@Vector(4, std.meta.Child(@TypeOf(from))){ v[0], v[1], v[2], w });
}

/// Returns the twist component of quaternion `q` about `axis` (the twist part of a swing-twist
/// decomposition). `axis` must be normalized. Ported from Jolt's `Quat::GetTwist`.
pub fn get_twist(q: anytype, axis: @Vector(3, std.meta.Child(@TypeOf(q)))) @Vector(4, std.meta.Child(@TypeOf(q))) {
    comptime {
        std.debug.assert(@typeInfo(@TypeOf(q)) == .vector);
        std.debug.assert(@typeInfo(@TypeOf(q)).vector.len == 4);
    }
    const T = std.meta.Child(@TypeOf(q));
    const q_xyz = @Vector(3, T){ q[0], q[1], q[2] };
    // Project the vector part onto the axis; keep the scalar part.
    const proj = @as(@Vector(3, T), @splat(vector.dot(q_xyz, axis))) * axis;
    const twist: @Vector(4, T) = .{ proj[0], proj[1], proj[2], q[3] };
    const twist_len = vector.norm_sqr(twist);
    if (twist_len != 0.0) {
        return twist / @as(@Vector(4, T), @splat(std.math.sqrt(twist_len)));
    }
    return identity(T);
}

//TODO: optimize with simd
pub fn mul(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    comptime {
        std.debug.assert(@typeInfo(@TypeOf(a)) == .vector);
        std.debug.assert(@typeInfo(@TypeOf(a)).vector.len == 4);
    }
    const inner_a = map_to_vector(a);
    const inner_b = map_to_vector(b);

    const lx = inner_a[0];
    const ly = inner_a[1];
    const lz = inner_a[2];
    const lw = inner_a[3];

    const rx = inner_b[0];
    const ry = inner_b[1];
    const rz = inner_b[2];
    const rw = inner_b[3];

    return @TypeOf(a){ lw * rx + lx * rw + ly * rz - lz * ry, lw * ry - lx * rz + ly * rw + lz * rx, lw * rz + lx * ry - ly * rx + lz * rw, lw * rw - lx * rx - ly * ry - lz * rz };
}

pub fn norm(q: anytype) std.meta.Float(@bitSizeOf(std.meta.Child(@TypeOf(q)))) {
    return vector.norm(q);
}

pub fn to_axis_angle(q: anytype) struct {
    axis: @Vector(3, std.meta.Child(@TypeOf(q))),
    angle: std.meta.Child(@TypeOf(q)),
} {
    comptime {
        std.debug.assert(@typeInfo(@TypeOf(q)) == .vector);
        std.debug.assert(@typeInfo(@TypeOf(q)).vector.len == 4);
    }
    const C = std.meta.Child(@TypeOf(q));
    const qw_clamped = std.math.clamp(q[3], -1.0, 1.0);
    const angle = 2.0 * std.math.acos(qw_clamped);
    const s = std.math.sqrt(1.0 - qw_clamped * qw_clamped);
    if (s < 0.001) { // axis direction is arbitrary when the angle is close to zero
        return .{ .axis = .{ 1, 0, 0 }, .angle = angle };
    }
    return .{ .axis = @Vector(3, C){ q[0] / s, q[1] / s, q[2] / s }, .angle = angle };
}

/// Rotate a 3D vector by a (unit) quaternion. Uses the branchless
/// t = 2·(u × v); v' = v + w·t + u × t form (u = q.xyz).
pub fn rotate_vector(q: anytype, v: @Vector(3, std.meta.Child(@TypeOf(q)))) @Vector(3, std.meta.Child(@TypeOf(q))) {
    comptime {
        std.debug.assert(@typeInfo(@TypeOf(q)) == .vector);
        std.debug.assert(@typeInfo(@TypeOf(q)).vector.len == 4);
    }
    const C = std.meta.Child(@TypeOf(q));
    const u = @Vector(3, C){ q[0], q[1], q[2] };
    const t = @as(@Vector(3, C), @splat(2)) * vector.cross(u, v);
    return v + @as(@Vector(3, C), @splat(q[3])) * t + vector.cross(u, t);
}

/// Rotate a 3D vector by the inverse of a (unit) quaternion.
pub fn rotate_vector_inv(q: anytype, v: @Vector(3, std.meta.Child(@TypeOf(q)))) @Vector(3, std.meta.Child(@TypeOf(q))) {
    return rotate_vector(conjugate(q), v);
}

/// Convert a quaternion (x, y, z, w) to a 4x4 rotation matrix.
pub fn to_matrix(q: anytype) zml.Mat(std.meta.Child(@TypeOf(q)), 4, 4) {
    comptime {
        std.debug.assert(@typeInfo(@TypeOf(q)) == .vector);
        std.debug.assert(@typeInfo(@TypeOf(q)).vector.len == 4);
    }
    return zml.Mat(std.meta.Child(@TypeOf(q)), 4, 4).from_quat(q);
}

/// Extract a quaternion from the rotation part of a matrix (Shepperd's method).
pub fn from_matrix(m: anytype) @Vector(4, @TypeOf(m).Type) {
    // element(row, col) is stored column-major as items[col][row]
    const m00 = m.items[0][0];
    const m01 = m.items[1][0];
    const m02 = m.items[2][0];
    const m10 = m.items[0][1];
    const m11 = m.items[1][1];
    const m12 = m.items[2][1];
    const m20 = m.items[0][2];
    const m21 = m.items[1][2];
    const m22 = m.items[2][2];
    const trace = m00 + m11 + m22;
    if (trace > 0) {
        const s = @sqrt(trace + 1.0) * 2.0; // s = 4 * qw
        return .{ (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s };
    } else if (m00 > m11 and m00 > m22) {
        const s = @sqrt(1.0 + m00 - m11 - m22) * 2.0; // s = 4 * qx
        return .{ 0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s };
    } else if (m11 > m22) {
        const s = @sqrt(1.0 + m11 - m00 - m22) * 2.0; // s = 4 * qy
        return .{ (m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s };
    } else {
        const s = @sqrt(1.0 + m22 - m00 - m11) * 2.0; // s = 4 * qz
        return .{ (m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s };
    }
}

pub fn conjugate(q: anytype) @TypeOf(q) {
    comptime {
        std.debug.assert(@typeInfo(@TypeOf(q)) == .vector);
        std.debug.assert(@typeInfo(@TypeOf(q)).vector.len == 4);
    }
    return q * @Vector(4, std.meta.Child(@TypeOf(q))){ -1, -1, -1, 1 };
}

pub fn inverse(q: anytype) @TypeOf(q) {
    comptime {
        std.debug.assert(@typeInfo(@TypeOf(q)) == .vector);
        std.debug.assert(@typeInfo(@TypeOf(q)).vector.len == 4);
    }
    return conjugate(q) / @as(@TypeOf(q), @splat(vector.norm(q)));
}

/// Returns `q` scaled to unit length, or `identity` when `q` is too short to denote a
/// rotation. `vector.normalize` divides by the norm unconditionally, so a zero quaternion
/// yields NaN; a rotation wants a usable value instead.
///
/// The threshold is on the *squared* norm, so no square root is taken on the reject path.
pub fn normalize(q: anytype, min_norm_sqr: std.meta.Child(@TypeOf(q))) @TypeOf(q) {
    comptime {
        std.debug.assert(@typeInfo(@TypeOf(q)) == .vector);
        std.debug.assert(@typeInfo(@TypeOf(q)).vector.len == 4);
    }
    const T = std.meta.Child(@TypeOf(q));
    const len_sqr = vector.norm_sqr(q);
    if (len_sqr <= min_norm_sqr) return identity(T);
    return q / @as(@TypeOf(q), @splat(@sqrt(len_sqr)));
}

/// `normalize` with a default threshold of 1.0e-6 on the squared norm.
pub fn normalize_default(q: anytype) @TypeOf(q) {
    return normalize(q, 1.0e-6);
}

pub fn slerp(a: anytype, b: anytype, factor: std.meta.Child(@TypeOf(a))) Quat(std.meta.Child(@TypeOf(a))) {
    comptime {
        std.debug.assert(@typeInfo(@TypeOf(a)) == .vector);
        std.debug.assert(@typeInfo(@TypeOf(a)).vector.len == 4);
        std.debug.assert(@typeInfo(@TypeOf(b)) == .vector);
        std.debug.assert(@typeInfo(@TypeOf(b)).vector.len == 4);
        std.debug.assert(@TypeOf(a) == @TypeOf(b));
    }
    const inner_a = map_to_vector(a);
    const inner_b = map_to_vector(b);
    const delta: std.meta.Child(@TypeOf(a)) = 0.0001;

    var sign_scale1: std.meta.Child(@TypeOf(a)) = 1.0;
    var cos_omega = vector.dot(inner_a, inner_b);

    if (cos_omega < 0.0) {
        cos_omega = -cos_omega;
        sign_scale1 = -1.0;
    }

    // Calculate coefficients
    var scale0: std.meta.Child(@TypeOf(a)) = undefined;
    var scale1: std.meta.Child(@TypeOf(a)) = undefined;
    if (1.0 - cos_omega > delta) {
        // Standard case (slerp)
        const omega = std.math.acos(cos_omega);
        const sin_omega = std.math.sin(omega);
        scale0 = std.math.sin((1.0 - factor) * omega) / sin_omega;
        scale1 = sign_scale1 * std.math.sin(factor * omega) / sin_omega;
    } else {
        // Quaternions are very close so we can do a linear interpolation
        scale0 = 1.0 - factor;
        scale1 = sign_scale1 * factor;
    }

    return vector.normalize(@as(Quat(std.meta.Child(@TypeOf(a))), @splat(scale0)) * inner_a +
        @as(Quat(std.meta.Child(@TypeOf(a))), @splat(scale1)) * inner_b);
}

pub fn lerp(a: anytype, b: anytype, factor: std.meta.Child(@TypeOf(a))) Quat(std.meta.Child(@TypeOf(a))) {
    comptime {
        std.debug.assert(@typeInfo(@TypeOf(a)) == .vector);
        std.debug.assert(@typeInfo(@TypeOf(a)).vector.len == 4);
        std.debug.assert(@typeInfo(@TypeOf(b)) == .vector);
        std.debug.assert(@typeInfo(@TypeOf(b)).vector.len == 4);
        std.debug.assert(@TypeOf(a) == @TypeOf(b));
    }

    return @as(Quat(std.meta.Child(@TypeOf(a))), @splat(1.0 - factor)) * map_to_vector(a) +
        @as(Quat(std.meta.Child(@TypeOf(a))), @splat(factor)) * map_to_vector(b);
}

/// Normalized lerp: interpolates along the chord and renormalizes, taking the shorter of the
/// two arcs. Cheaper than `slerp` but not constant angular velocity.
///
/// Unlike `lerp`, this negates `b` when the two quaternions lie on opposite hemispheres, so
/// `nlerp(q, -q, t)` stays at `q` rather than sweeping through the antipodal rotation. Since
/// `q` and `-q` denote the same orientation, that flip is what makes the result track the
/// intended rotation.
pub fn nlerp(a: anytype, b: anytype, factor: std.meta.Child(@TypeOf(a))) Quat(std.meta.Child(@TypeOf(a))) {
    comptime {
        std.debug.assert(@typeInfo(@TypeOf(a)) == .vector);
        std.debug.assert(@typeInfo(@TypeOf(a)).vector.len == 4);
        std.debug.assert(@typeInfo(@TypeOf(b)) == .vector);
        std.debug.assert(@typeInfo(@TypeOf(b)).vector.len == 4);
        std.debug.assert(@TypeOf(a) == @TypeOf(b));
    }
    const inner_a = map_to_vector(a);
    const inner_b = map_to_vector(b);
    const nearest = if (vector.dot(inner_a, inner_b) < 0) -inner_b else inner_b;
    return normalize_default(lerp(inner_a, nearest, factor));
}

/// Intrinsic Z-Y-X (equivalently extrinsic X-Y-Z): `q = qz * qy * qx`, i.e. the X rotation is
/// applied first about the fixed axes. `angles` is ordered `.{ x, y, z }` regardless.
///
/// See `from_euler_xyz_intrinsic` for the mirrored convention.
pub fn from_eular_angles(inAngles: anytype) @Vector(4, std.meta.Child(@TypeOf(inAngles))) {
    comptime {
        std.debug.assert(@typeInfo(@TypeOf(inAngles)) == .vector);
        std.debug.assert(@typeInfo(@TypeOf(inAngles)).vector.len == 3);
    }

    const half = @as(@TypeOf(inAngles), @splat(0.5)) * inAngles;
    const res = vector.sin_cos(half);

    const cx = res.cos_out[0];
    const sx = res.sin_out[0];
    const cy = res.cos_out[1];
    const sy = res.sin_out[1];
    const cz = res.cos_out[2];
    const sz = res.sin_out[2];

    return .{ cz * sx * cy - sz * cx * sy, cz * cx * sy + sz * sx * cy, sz * cx * cy - cz * sx * sy, cz * cx * cy + sz * sx * sy };
}

pub fn from_rotation(axis: anytype, angle: std.meta.Child(@TypeOf(axis))) @Vector(4, std.meta.Child(@TypeOf(axis))) {
    comptime {
        std.debug.assert(@typeInfo(@TypeOf(axis)) == .vector);
        std.debug.assert(@typeInfo(@TypeOf(axis)).vector.len == 3);
    }
    const in_axis = map_to_vector(axis);
    std.debug.assert(vector.is_normalized_default(in_axis));
    return .{ in_axis[0] * std.math.sin(angle * 0.5), in_axis[1] * std.math.sin(angle * 0.5), in_axis[2] * std.math.sin(angle * 0.5), std.math.cos(angle * 0.5) };
}

/// Inverse of `from_eular_angles`: recovers intrinsic Z-Y-X angles as `.{ x, y, z }`.
/// `q` is assumed normalized.
pub fn to_eular_angles(q: anytype) @Vector(3, std.meta.Child(@TypeOf(q))) {
    comptime {
        std.debug.assert(@typeInfo(@TypeOf(q)) == .vector);
        std.debug.assert(@typeInfo(@TypeOf(q)).vector.len == 4);
    }

    const ysqr = q[1] * q[1];

    // roll (x-axis rotation)
    const t0 = 2.0 * (q[3] * q[0] + q[1] * q[2]);
    const t1 = 1.0 - 2.0 * (q[0] * q[0] + ysqr);
    const roll = std.math.atan2(t0, t1);

    // pitch (y-axis rotation)
    var t2 = 2.0 * (q[3] * q[1] - q[2] * q[0]);
    t2 = if (t2 > 1.0) 1.0 else if (t2 < -1.0) -1.0 else t2;
    const pitch = std.math.asin(t2);

    // yaw (z-axis rotation)
    const t3 = 2.0 * (q[3] * q[2] + q[0] * q[1]);
    const t4 = 1.0 - 2.0 * (ysqr + q[2] * q[2]);
    const yaw = std.math.atan2(t3, t4);

    return .{ roll, pitch, yaw };
}

/// Intrinsic X-Y-Z (equivalently extrinsic Z-Y-X): `q = qx * qy * qz`, i.e. the Z rotation is
/// applied first about the fixed axes. `angles` is ordered `.{ x, y, z }`.
///
/// This is the mirror of `from_eular_angles`, which composes in the opposite order. The two
/// are not interchangeable; pick by the convention your data was authored in.
pub fn from_euler_xyz_intrinsic(angles: anytype) @Vector(4, std.meta.Child(@TypeOf(angles))) {
    comptime {
        std.debug.assert(@typeInfo(@TypeOf(angles)) == .vector);
        std.debug.assert(@typeInfo(@TypeOf(angles)).vector.len == 3);
    }

    const half = @as(@TypeOf(angles), @splat(0.5)) * angles;
    const res = vector.sin_cos(half);

    const sx = res.sin_out[0];
    const cx = res.cos_out[0];
    const sy = res.sin_out[1];
    const cy = res.cos_out[1];
    const sz = res.sin_out[2];
    const cz = res.cos_out[2];

    return .{
        sx * cy * cz + cx * sy * sz,
        cx * sy * cz - sx * cy * sz,
        cx * cy * sz + sx * sy * cz,
        cx * cy * cz - sx * sy * sz,
    };
}

/// Inverse of `from_euler_xyz_intrinsic`: recovers intrinsic X-Y-Z angles as `.{ x, y, z }`.
/// `q` is assumed normalized.
///
/// The Y angle is extracted through `asin` and so is clamped to [-pi/2, pi/2]; at the poles
/// (|sin y| == 1) the X and Z angles are not separable and the decomposition is not unique.
pub fn to_euler_xyz_intrinsic(q: anytype) @Vector(3, std.meta.Child(@TypeOf(q))) {
    comptime {
        std.debug.assert(@typeInfo(@TypeOf(q)) == .vector);
        std.debug.assert(@typeInfo(@TypeOf(q)).vector.len == 4);
    }
    const T = std.meta.Child(@TypeOf(q));

    // Matrix elements of the rotation q denotes: r12, r22, r02, r01, r00.
    const sin_x = 2.0 * (q[3] * q[0] - q[1] * q[2]);
    const cos_x = 1.0 - 2.0 * (q[0] * q[0] + q[1] * q[1]);
    const sin_y = std.math.clamp(2.0 * (q[3] * q[1] + q[2] * q[0]), @as(T, -1.0), @as(T, 1.0));
    const sin_z = 2.0 * (q[3] * q[2] - q[0] * q[1]);
    const cos_z = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]);

    return .{
        std.math.atan2(sin_x, cos_x),
        std.math.asin(sin_y),
        std.math.atan2(sin_z, cos_z),
    };
}

test from_to {
    try std.testing.expect(vector.is_close_default(from_to(@Vector(3, f32){ 10, 0, 0 }, @Vector(3, f32){ 20, 0, 0 }), identity(f32)));
}

test mul {
    try std.testing.expect(vector.is_close_default(mul(Quat4f32{ 0, 1, 0, 0 }, Quat4f32{ 1, 0, 0, 0 }), Quat4f32{ 0, 0, -1, 0 }));
    try std.testing.expect(vector.is_close_default(mul(Quat4f32{ 1, 0, 0, 0 }, Quat4f32{ 0, 1, 0, 0 }), Quat4f32{ 0, 0, 1, 0 }));
    try std.testing.expect(vector.is_close_default(mul(Quat4f32{ 2, 3, 4, 1 }, Quat4f32{ 6, 7, 8, 5 }), Quat4f32{ 12, 30, 24, -60 }));
}

test slerp {
    const v1 = identity(f32);
    const v2: Quat4f32 = from_rotation(zml.Vec3f32{ 1, 0, 0 }, 0.99 * std.math.pi);
    try std.testing.expect(vector.is_close_default(slerp(v1, v2, 0.25), from_rotation(x_axis(f32), 0.25 * 0.99 * std.math.pi)));

    const v3 = vector.normalize(Quat4f32{ 1, 2, 3, 4 });
    try std.testing.expect(vector.is_close_default(slerp(v3, -v3, 0.5), v3));
}

test lerp {
    const v1: Quat4f32 = .{ 1, 2, 3, 4 };
    const v2: Quat4f32 = .{ 5, 6, 7, 8 };
    try std.testing.expect(vector.is_close_default(lerp(v1, v2, 0.25), Quat4f32{ 2, 3, 4, 5 }));
}

test normalize {
    // A degenerate quaternion falls back to identity rather than dividing through to NaN.
    try std.testing.expectEqual(identity(f32), normalize_default(Quat4f32{ 0, 0, 0, 0 }));
    try std.testing.expect(std.math.isNan(vector.normalize(Quat4f32{ 0, 0, 0, 0 })[0]));

    // An already-unit quaternion is left alone.
    const unit = from_rotation(x_axis(f32), 0.7);
    try std.testing.expect(vector.is_close(normalize_default(unit), unit, 1e-12));

    // A scaled one is brought back to unit length.
    const scaled = @as(Quat4f32, @splat(3.0)) * unit;
    try std.testing.expect(vector.is_close(normalize_default(scaled), unit, 1e-12));
    try std.testing.expect(vector.is_normalized_default(normalize_default(Quat4f32{ 1, 2, 3, 4 })));

    // The threshold is on the squared norm.
    const tiny = @as(Quat4f32, @splat(1.0e-4)) * unit; // norm_sqr == 1.0e-8
    try std.testing.expectEqual(identity(f32), normalize_default(tiny));
    try std.testing.expect(vector.is_close(normalize(tiny, 1.0e-12), unit, 1e-6));
}

test nlerp {
    const a = from_rotation(x_axis(f32), 0.3);
    const b = from_rotation(x_axis(f32), 1.1);

    // Endpoints are reproduced, and every sample stays on the unit sphere.
    try std.testing.expect(vector.is_close(nlerp(a, b, 0), a, 1e-6));
    try std.testing.expect(vector.is_close(nlerp(a, b, 1), b, 1e-6));
    for ([_]f32{ 0, 0.25, 0.5, 0.75, 1 }) |t| {
        try std.testing.expect(vector.is_normalized_default(nlerp(a, b, t)));
    }

    // Along a single axis nlerp stays on the arc, so it agrees with slerp.
    try std.testing.expect(vector.is_close(nlerp(a, b, 0.5), slerp(a, b, 0.5), 1e-6));

    // The hemisphere flip: q and -q denote the same orientation, so interpolating between
    // them must stay put. Plain `lerp` collapses to zero at the midpoint instead.
    try std.testing.expect(vector.is_close(nlerp(a, -a, 0.5), a, 1e-6));
    try std.testing.expect(vector.is_close_default(lerp(a, -a, 0.5), @as(Quat4f32, @splat(0))));
}

test from_eular_angles {
    // Pins the convention: intrinsic Z-Y-X, i.e. qz * qy * qx.
    const angles = @Vector(3, f32){ 0.3, -0.7, 1.1 };
    const composed = mul(
        from_rotation(z_axis(f32), angles[2]),
        mul(from_rotation(y_axis(f32), angles[1]), from_rotation(x_axis(f32), angles[0])),
    );
    try std.testing.expect(vector.is_close(from_eular_angles(angles), composed, 1e-6));

    // Single-axis inputs reduce to the corresponding axis rotation.
    try std.testing.expect(vector.is_close(
        from_eular_angles(@Vector(3, f32){ 0.4, 0, 0 }),
        from_rotation(x_axis(f32), 0.4),
        1e-6,
    ));
    try std.testing.expect(vector.is_close(
        from_eular_angles(@Vector(3, f32){ 0, 0, -0.9 }),
        from_rotation(z_axis(f32), -0.9),
        1e-6,
    ));
}

test to_eular_angles {
    // Round trips away from the gimbal-lock poles.
    const angles = @Vector(3, f32){ 0.3, -0.7, 1.1 };
    try std.testing.expect(vector.is_close(to_eular_angles(from_eular_angles(angles)), angles, 1e-6));
}

test from_euler_xyz_intrinsic {
    // Pins the mirrored convention: intrinsic X-Y-Z, i.e. qx * qy * qz.
    const angles = @Vector(3, f32){ 0.3, -0.7, 1.1 };
    const composed = mul(
        from_rotation(x_axis(f32), angles[0]),
        mul(from_rotation(y_axis(f32), angles[1]), from_rotation(z_axis(f32), angles[2])),
    );
    try std.testing.expect(vector.is_close(from_euler_xyz_intrinsic(angles), composed, 1e-6));

    // The result is unit length without an explicit normalize.
    try std.testing.expect(vector.is_normalized_default(from_euler_xyz_intrinsic(angles)));

    // Single-axis inputs agree with `from_eular_angles`; multi-axis ones must not.
    try std.testing.expect(vector.is_close(
        from_euler_xyz_intrinsic(@Vector(3, f32){ 0, 0.4, 0 }),
        from_eular_angles(@Vector(3, f32){ 0, 0.4, 0 }),
        1e-6,
    ));
    try std.testing.expect(!vector.is_close(
        from_euler_xyz_intrinsic(angles),
        from_eular_angles(angles),
        1e-6,
    ));
}

test to_euler_xyz_intrinsic {
    // Round trips away from the gimbal-lock poles.
    const angles = @Vector(3, f32){ 0.3, -0.7, 1.1 };
    try std.testing.expect(vector.is_close(
        to_euler_xyz_intrinsic(from_euler_xyz_intrinsic(angles)),
        angles,
        1e-6,
    ));

    // Recovering from a composed rotation gives back the generating angles.
    for ([_]@Vector(3, f32){
        .{ 0, 0, 0 },
        .{ 0.5, 0, 0 },
        .{ 0, 0, -1.2 },
        .{ -0.9, 1.0, 0.2 },
    }) |e| {
        try std.testing.expect(vector.is_close(
            to_euler_xyz_intrinsic(from_euler_xyz_intrinsic(e)),
            e,
            1e-6,
        ));
    }
}

test rotate_vector {
    const q = from_rotation(z_axis(f32), std.math.pi / 2.0);
    // 90° about +z takes +x to +y (right-handed, active rotation).
    try std.testing.expect(vector.is_close(rotate_vector(q, @Vector(3, f32){ 1, 0, 0 }), .{ 0, 1, 0 }, 1e-6));
    // inverse rotation undoes it
    const v = @Vector(3, f32){ 0.3, 0.5, -0.2 };
    try std.testing.expect(vector.is_close(rotate_vector_inv(q, rotate_vector(q, v)), v, 1e-6));
}

test to_matrix {
    const q = vector.normalize(from_eular_angles(@Vector(3, f32){ 0.3, -0.6, 1.1 }));
    const m = to_matrix(q);
    const v = @Vector(3, f32){ 0.2, -0.7, 0.5 };
    // Applying the matrix (as a column vector) must match rotate_vector.
    var mv: @Vector(3, f32) = .{ 0, 0, 0 };
    inline for (0..3) |c| {
        // column c of the (column-major) matrix, first three components
        mv += @Vector(3, f32){ m.items[c][0], m.items[c][1], m.items[c][2] } * @as(@Vector(3, f32), @splat(v[c]));
    }
    try std.testing.expect(vector.is_close(mv, rotate_vector(q, v), 1e-5));
}

test from_matrix {
    const q = vector.normalize(from_eular_angles(@Vector(3, f32){ 0.3, -0.6, 1.1 }));
    var back = from_matrix(to_matrix(q));
    // q and -q are the same rotation; align signs before comparing.
    if (vector.dot(q, back) < 0) back = -back;
    try std.testing.expect(vector.is_close(back, q, 1e-5));
}

// The round-trip above only pins `from_matrix` transitively through `to_matrix`; a storage
// convention change that transposed both consistently would leave it green. This pins
// `from_matrix`'s storage reads against an explicitly laid-out, asymmetric rotation matrix.
test "from_matrix known rotation" {
    // 90° about +z (column-vector, active), written row by row: sends +x -> +y, +y -> -x.
    const m: zml.Mat(f32, 4, 4) = .from_rows(.{
        .{ 0, -1, 0, 0 },
        .{ 1, 0, 0, 0 },
        .{ 0, 0, 1, 0 },
        .{ 0, 0, 0, 1 },
    });
    const q = from_matrix(m);
    try std.testing.expect(vector.is_close(rotate_vector(q, .{ 1, 0, 0 }), .{ 0, 1, 0 }, 1e-5));
    try std.testing.expect(vector.is_close(rotate_vector(q, .{ 0, 1, 0 }), .{ -1, 0, 0 }, 1e-5));
}

test to_axis_angle {
    const axis = vector.normalize(@Vector(3, f32){ 1, 2, 3 });
    const angle: f32 = 1.2;
    const aa = to_axis_angle(from_rotation(axis, angle));
    try std.testing.expectApproxEqAbs(angle, aa.angle, 1e-5);
    try std.testing.expect(vector.is_close(aa.axis, axis, 1e-5));
}

test get_twist {
    // A rotation purely about the query axis is entirely twist -> returns itself.
    const qz = from_rotation(z_axis(f32), 0.7);
    try std.testing.expect(vector.is_close(get_twist(qz, .{ 0, 0, 1 }), qz, 1e-5));
    // A rotation about a perpendicular axis has no twist about +z -> identity.
    const qx = from_rotation(x_axis(f32), 0.7);
    try std.testing.expect(vector.is_close(get_twist(qx, .{ 0, 0, 1 }), identity(f32), 1e-5));
}
