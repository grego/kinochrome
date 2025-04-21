use std::array;

use serde::{Deserialize, Serialize};
use vulkano::padded::Padded;

use crate::gpu_compute::PushConstantData;

/// Standard D65 illuminant in xy chromaticity coordinates.
const D65: [f32; 2] = [0.31272, 0.32903];
/// Standard D50 illuminant
#[allow(unused)]
const D50: [f32; 2] = [0.34567, 0.35850];

/// Standard A illuminant in xy chromaticity coordinates.
const A_ILLUM: [f32; 2] = [0.44758, 0.40745];

/// Matrix in column major order, since that's what shaders take.
type Mat<const N: usize> = [[f32; N]; N];

/// Parameters for color grading
#[derive(Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct ColorParams {
    /// Camera matrix
    pub cam_matrix: Mat<3>,
    /// Illuminant - the light source
    pub illuminant: Illuminant,
}

/// Color of the light source
#[derive(Clone, Copy, PartialEq, Deserialize, Serialize)]
pub enum Illuminant {
    /// Standard tungsten
    A,
    /// Standard daylinght
    D(u16),
    /// Planckian blackbody radiation
    Blackbody(u16),
    /// Custom illuminant in Kirk Lch space
    Custom([f32; 3]),
}

impl Default for ColorParams {
    fn default() -> Self {
        ColorParams {
            cam_matrix: identity_mat(),
            illuminant: Illuminant::D(6500),
        }
    }
}

impl ColorParams {
    /// Set the compute shader push constants using these parameters
    pub fn set_push_constants(&self, pc: &PushConstantData) -> PushConstantData {
        let mut pc = *pc;
        pc.cam_matrix = self.adaptation_matrix().map(Padded);
        pc
    }

    /// Update the push constants if the color parameters were changed
    pub fn update_push_constants(&self, prev: &Self, pc: &mut PushConstantData) {
        if self.illuminant != prev.illuminant {
            pc.cam_matrix = self.adaptation_matrix().map(Padded);
        }
    }

    /// Return the CAT16 adaptation matrix for this illuminant
    pub fn adaptation_matrix(&self) -> Mat<3> {
        matmul(
            cat16_adaptation(self.illuminant.xy_coords(), D65),
            self.cam_matrix,
        )
    }

    /// Get xy chromaticity coordinates of a color in the camera RGB space
    pub const fn rgb_to_xy(&self, rgb: [f32; 3]) -> [f32; 2] {
        xyz_to_xy(matvecmul(self.cam_matrix, rgb))
    }
}

impl Illuminant {
    /// Make an array of possible illuminants, each with a default value.
    pub const fn defaults() -> [Self; 4] {
        [
            Illuminant::A,
            Illuminant::D(6500),
            Illuminant::Blackbody(6500),
            Illuminant::Custom([1.0, 0.0, 0.0]),
        ]
    }

    fn xy_coords(&self) -> [f32; 2] {
        match *self {
            Illuminant::A => A_ILLUM,
            Illuminant::D(t) => d_illuminant_chromaticity(t),
            Illuminant::Blackbody(t) => blackbody_chromaticity(t),
            Illuminant::Custom(ch) => ch_to_xy(ch),
        }
    }

    /// Color of the illuminant in the SRGB space
    pub fn srgb(&self) -> [f32; 3] {
        matvecmul(XYZ_TO_REC709, xy_to_xyz(self.xy_coords()))
    }

    /// Set the custom xy coordinates of the illuminant
    pub fn custom(xy: [f32; 2]) -> Self {
        Illuminant::Custom(xy_to_ch(xy))
    }

    /// Description of the illuminant
    pub const fn description(&self) -> &'static str {
        match self {
            Illuminant::A => "A (incandescent)",
            Illuminant::D(_) => "D (daylight)",
            Illuminant::Blackbody(_) => "Planckian (blackbody)",
            Illuminant::Custom(_) => "Custom",
        }
    }
}

/// Return CAT16 color adaptation matrix from XYZ to XYZ
const fn cat16_adaptation(src: [f32; 2], dst: [f32; 2]) -> Mat<3> {
    let src = matvecmul(M16, xy_to_xyz(src));
    let dst = matvecmul(M16, xy_to_xyz(dst));
    let mut ratio = [0.0; 3];
    let mut i = 0;
    while i < 3 {
        ratio[i] = dst[i] / src[i];
        i += 1;
    }
    matmul(M16I, matmul(diag_mat(ratio), M16))
}

/// Calculate the D illuminant chromaticity in xy chromaticity coordinates.
const fn d_illuminant_chromaticity(t: u16) -> [f32; 2] {
    let t = t as f32;
    let x = 0.244063 + 0.09911e+3 / t + 2.9678e+6 / (t * t) - 4.6070e+9 / (t * t * t);
    let y = -3.0 * x * x + 2.870 * x - 0.275;
    [x, y]
}

/// Calculate the blackbody chromaticity in xy chromaticity coordinates.
const fn blackbody_chromaticity(t: u16) -> [f32; 2] {
    let t = t as f32;
    let (t2, t3) = (t * t, t * t * t);
    let x = if t < 4000.0 {
        -2.66124e+8 / t3 - 2.34359e+5 / t2 + 8.77696e+2 / t + 0.179910
    } else {
        -3.02585e+9 / t3 + 2.10704e+6 / t2 + 2.22635e+2 / t + 0.240390
    };
    let (x2, x3) = (x * x, x * x * x);
    let y = if t < 2222.0 {
        -1.1063814 * x3 - 1.34811 * x2 + 2.1855583 * x - 0.2021968
    } else if t < 4000.0 {
        -0.9549476 * x3 - 1.3741859 * x2 + 2.09137 * x - 0.1674887
    } else {
        3.081758 * x3 - 5.873387 * x2 + 3.75113 * x - 0.3700148
    };
    [x, y]
}

/// Chromaticity coordinates with Y = 1.0
const fn xy_to_xyz([x, y]: [f32; 2]) -> [f32; 3] {
    [x / y, 1.0, (1.0 - x - y) / y]
}

/// Chromaticity coordinates with Y = 1.0
const fn xyz_to_xy([x, y, z]: [f32; 3]) -> [f32; 2] {
    let s = x + y + z;
    [x / s, y / s]
}

/// Chromaticity coordinates to Kirk Lch space
fn xy_to_ch(xy: [f32; 2]) -> [f32; 3] {
    let lms = matvecmul(XYZ_D65_TO_LMS_2006_D65, xy_to_xyz(xy));
    let y = 0.6899027 * lms[0] + 0.34832189 * lms[1];
    let s = lms[0] + lms[1] + lms[2];
    let lms = lms.map(|x| x / s);
    let r = 1.0671 * lms[0] - 0.6873 * lms[1] + 0.02062 - 0.21902143;
    let g = -0.0362 * lms[0] + 1.7182 * lms[1] - 0.05155 - 0.543714;
    [(r * r + g * g).sqrt(), r.atan2(g), y]
}

/// Kirk Lch to chromaticity coordinates
fn ch_to_xy([c, h, y]: [f32; 3]) -> [f32; 2] {
    let r = c * h.sin() + 0.21902143;
    let g = c * h.cos() + 0.543714;
    let l = 0.95 * r + 0.38 * g;
    let m = 0.02 * r + 0.59 * g + 0.03;
    let a = y / (0.6899027 * l + 0.34832189 * m);
    let lms = [l * a, m * a, (1.0 - l - m) * a];
    xyz_to_xy(matvecmul(LMS_2006_D65_TO_XYZ_D65, lms))
}

// Numerically stable 2x2 determinant
fn det2(a: f32, b: f32, c: f32, d: f32) -> f32 {
    let bc = b * c;
    let err = (-b).mul_add(c, bc);
    let det = a.mul_add(d, -bc);
    det + err
}

/// Inverse of a 3x3 matrix
pub fn inv3(m: Mat<3>) -> Mat<3> {
    let inv = array::from_fn(|i| {
        array::from_fn(|j| {
            det2(
                m[(j + 1) % 3][(i + 1) % 3],
                m[(j + 1) % 3][(i + 2) % 3],
                m[(j + 2) % 3][(i + 1) % 3],
                m[(j + 2) % 3][(i + 2) % 3],
            )
        })
    });
    let det = m[0][0] * inv[0][0] + m[0][1] * inv[1][0] + m[0][2] * inv[2][0];
    inv.map(|col| col.map(|a| a / det))
}

const fn matvecmul<const N: usize>(m: Mat<N>, v: [f32; N]) -> [f32; N] {
    let mut r = [0.0; N];
    let mut i = 0;
    while i < N {
        let mut j = 0;
        while j < N {
            r[i] += m[j][i] * v[j];
            j += 1;
        }
        i += 1;
    }
    r
}

const fn matmul<const N: usize>(m: Mat<N>, n: Mat<N>) -> Mat<N> {
    let mut r = [[0.0; N]; N];
    let mut i = 0;
    while i < N {
        let mut j = 0;
        while j < N {
            let mut k = 0;
            while k < N {
                r[i][j] += m[k][j] * n[i][k];
                k += 1;
            }
            j += 1;
        }
        i += 1;
    }
    r
}

/// Matrix transposition
pub const fn transpose<const N: usize>(m: Mat<N>) -> Mat<N> {
    let mut mat = [[0.0; N]; N];
    let mut i = 0;
    while i < N {
        let mut j = 0;
        while j < N {
            mat[i][j] = m[j][i];
            j += 1;
        }
        i += 1;
    }
    mat
}

/// Diagonal matrix
const fn diag_mat<const N: usize>(v: [f32; N]) -> Mat<N> {
    let mut mat = [[0.0; N]; N];
    let mut i = 0;
    while i < N {
        mat[i][i] = v[i];
        i += 1;
    }
    mat
}

/// Identity matrix
pub const fn identity_mat<const N: usize>() -> Mat<N> {
    diag_mat([1.0; N])
}

/// Interpret an array as 3x3 matrix
pub const fn array_to_mat3(a: [f32; 9]) -> Mat<3> {
    let mut mat = [[0.0; 3]; 3];
    let mut i = 0;
    while i < 3 {
        mat[i] = [a[3 * i], a[3 * i + 1], a[3 * i + 2]];
        i += 1;
    }
    mat
}

#[allow(unused)]
const BGR: Mat<3> = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];

const M16I: Mat<3> = transpose([
    [1.8620679, -1.0112546, 0.1491868],
    [0.3875265, 0.6214474, -0.008974],
    [-0.0158415, -0.0341229, 1.0499644],
]);

const M16: Mat<3> = transpose([
    [0.401288, 0.650173, -0.051461],
    [-0.250268, 1.204414, 0.045854],
    [-0.002079, 0.048952, 0.953127],
]);

#[allow(unused)]
const REC2020_TO_XYZ: Mat<3> = [
    [6.3695805e-1, 2.627002e-1, 4.2057587e-11],
    [1.446169e-1, 6.7799807e-1, 2.8072693e-2],
    [1.6888098e-1, 5.9301717e-2, 1.0609851],
];

const XYZ_TO_REC709: Mat<3> = [
    [3.2404542, -0.969266, 0.0556434],
    [-1.5371385, 1.8760108, -0.2040259],
    [-0.4985314, 0.0415560, 1.0572252],
];

const XYZ_D65_TO_LMS_2006_D65: Mat<3> = [
    [0.257085, -0.394427, 0.064856],
    [0.859943, 1.1758, -0.076250],
    [-0.031061, 0.106423, 0.559067],
];

const LMS_2006_D65_TO_XYZ_D65: Mat<3> = [
    [1.8079466, 0.6178396, -0.1254696],
    [-1.2997166, 0.3959545, 0.2047804],
    [0.3478588, -0.0410469, 1.7427418],
];
