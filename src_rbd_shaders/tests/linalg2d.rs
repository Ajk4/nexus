use crate::utils::linalg::*;
use glamx::{Rot2, Vec2};

#[test]
fn angular_jacobians_match_rotating_point_velocity() {
    let point = Vec2::new(1.3, -0.8);
    let angular = 0.7;
    let dt = 0.001;
    let expected = (Rot2::from_angle(angular * dt) * point
        - Rot2::from_angle(-angular * dt) * point) / (2.0 * dt);
    let c = MatSlice::dense(0, 2, 1);
    let b = MatSlice::dense(0, 1, 1);
    let mut serial = [0.0; 2];
    let mut parallel = serial;
    gemm_skew_tr_lhs_cross_buf(&mut serial, c, 1.0, point, &[angular], b, 0.0);
    gemm_skew_tr_lhs_cross_buf_par(&mut parallel, c, 1.0, point, &[angular], b, 0.0, 0, 1);
    for result in [serial, parallel] {
        assert!(Vec2::from(result).distance(expected) < 0.001);
    }
    let b = MatSlice::dense(2, 1, 1);
    let mut serial = [0.0, 0.0, angular];
    let mut parallel = serial;
    gemm_skew_tr_lhs(&mut serial, c, 1.0, point, b, 0.0);
    gemm_skew_tr_lhs_par(&mut parallel, c, 1.0, point, b, 0.0, 0, 1);
    for result in [serial, parallel] {
        assert!(Vec2::new(result[0], result[1]).distance(expected) < 0.001);
    }
    let b = MatSlice::dense(0, 1, 1);
    let mut serial = [0.0; 2];
    let mut parallel = serial;
    gemm_omega_skew_tr_cross_buf(&mut serial, c, 1.0, angular, point, &[angular], b, 0.0);
    gemm_omega_skew_tr_cross_buf_par(&mut parallel, c, 1.0, angular, point, &[angular], b, 0.0, 0, 1);
    // A point rotating at constant speed accelerates toward the center.
    let expected_acceleration = -point * angular * angular;
    for result in [serial, parallel] {
        assert!(Vec2::from(result).distance(expected_acceleration) < 1.0e-6);
    }
}
