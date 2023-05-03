import Foundation

protocol Spectrum {
        static func * (lhs: Self, rhs: Self) -> Self

        // Temporary: We adopt a step-by-step strategy to convert the renderer to spectral
        // rendering; first, convert the old Spectrum class to RGBSpectrum (done), then make
        // Spectrum a protocol (done), add additional classes like PiecewiseLinearSpectrum
        // (done) and facilities to convert to RGBSpectrum (here). After that a SampledSpectrum
        // class can be added and all computation converted to this one.
        func asRgb() -> RGBSpectrum
}

struct PiecewiseLinearSpectrum: Spectrum {

        private func findIndex(wavelength: FloatX) -> Int {
                for (index, lambda) in lambdas.enumerated() {
                        if wavelength < lambda {
                                return index
                        }
                }
                return lambdas.count - 1
        }

        func asRgb() -> RGBSpectrum {
                let red = values[findIndex(wavelength: 630)]
                let green = values[findIndex(wavelength: 532)]
                let blue = values[findIndex(wavelength: 465)]
                return RGBSpectrum(r: red, g: green, b: blue)
        }

        let lambdas: [FloatX]
        let values: [FloatX]
}

extension PiecewiseLinearSpectrum {
        static func * (lhs: Self, rhs: Self) -> Self {
                fatalError()
        }
}

let metalWavelengths: [FloatX] = [
        298.757050, 302.400421, 306.133759, 309.960449, 313.884003, 317.908142, 322.036835,
        326.274139, 330.624481, 335.092377, 339.682678, 344.400482, 349.251221, 354.240509,
        359.374420, 364.659332, 370.102020, 375.709625, 381.489777, 387.450562, 393.600555,
        399.948975, 406.505493, 413.280579, 420.285339, 427.531647, 435.032196, 442.800629,
        450.851562, 459.200653, 467.864838, 476.862213, 486.212463, 495.936707, 506.057861,
        516.600769, 527.592224, 539.061646, 551.040771, 563.564453, 576.670593, 590.400818,
        604.800842, 619.920898, 635.816284, 652.548279, 670.184753, 688.800964, 708.481018,
        729.318665, 751.419250, 774.901123, 799.897949, 826.561157, 855.063293, 885.601257,
]

// Silver refractive indices
let metalAgEtaSpectrum = PiecewiseLinearSpectrum(
        lambdas: metalWavelengths,
        values: [
                1.519000, 1.496000, 1.432500, 1.323000, 1.142062, 0.932000, 0.719062,
                0.526000, 0.388125, 0.294000, 0.253313, 0.238000, 0.221438, 0.209000,
                0.194813, 0.186000, 0.192063, 0.200000, 0.198063, 0.192000, 0.182000,
                0.173000, 0.172625, 0.173000, 0.166688, 0.160000, 0.158500, 0.157000,
                0.151063, 0.144000, 0.137313, 0.132000, 0.130250, 0.130000, 0.129938,
                0.130000, 0.130063, 0.129000, 0.124375, 0.120000, 0.119313, 0.121000,
                0.125500, 0.131000, 0.136125, 0.140000, 0.140063, 0.140000, 0.144313,
                0.148000, 0.145875, 0.143000, 0.142563, 0.145000, 0.151938, 0.163000,
        ])

let metalAlEtaSpectrum = PiecewiseLinearSpectrum(
        lambdas: metalWavelengths,
        values: [
                0.273375, 0.280000, 0.286813, 0.294000, 0.301875, 0.310000, 0.317875,
                0.326000, 0.334750, 0.344000, 0.353813, 0.364000, 0.374375, 0.385000,
                0.395750, 0.407000, 0.419125, 0.432000, 0.445688, 0.460000, 0.474688,
                0.490000, 0.506188, 0.523000, 0.540063, 0.558000, 0.577313, 0.598000,
                0.620313, 0.644000, 0.668625, 0.695000, 0.723750, 0.755000, 0.789000,
                0.826000, 0.867000, 0.912000, 0.963000, 1.020000, 1.080000, 1.150000,
                1.220000, 1.300000, 1.390000, 1.490000, 1.600000, 1.740000, 1.910000,
                2.140000, 2.410000, 2.630000, 2.800000, 2.740000, 2.580000, 2.240000,
        ])

// Copper refractive indices
let metalCuEtaSpectrum = PiecewiseLinearSpectrum(
        lambdas: metalWavelengths,
        values: [
                1.400313, 1.380000, 1.358438, 1.340000, 1.329063, 1.325000, 1.332500,
                1.340000, 1.334375, 1.325000, 1.317812, 1.310000, 1.300313, 1.290000,
                1.281563, 1.270000, 1.249062, 1.225000, 1.200000, 1.180000, 1.174375,
                1.175000, 1.177500, 1.180000, 1.178125, 1.175000, 1.172812, 1.170000,
                1.165312, 1.160000, 1.155312, 1.150000, 1.142812, 1.135000, 1.131562,
                1.120000, 1.092437, 1.040000, 0.950375, 0.826000, 0.645875, 0.468000,
                0.351250, 0.272000, 0.230813, 0.214000, 0.209250, 0.213000, 0.216250,
                0.223000, 0.236500, 0.250000, 0.254188, 0.260000, 0.280000, 0.300000,
        ])

// Silver extinction coefficients
let metalAgKSpectrum = PiecewiseLinearSpectrum(
        lambdas: metalWavelengths,
        values: [
                1.080000, 0.882000, 0.761063, 0.647000, 0.550875, 0.504000, 0.554375,
                0.663000, 0.818563, 0.986000, 1.120687, 1.240000, 1.345250, 1.440000,
                1.533750, 1.610000, 1.641875, 1.670000, 1.735000, 1.810000, 1.878750,
                1.950000, 2.029375, 2.110000, 2.186250, 2.260000, 2.329375, 2.400000,
                2.478750, 2.560000, 2.640000, 2.720000, 2.798125, 2.880000, 2.973750,
                3.070000, 3.159375, 3.250000, 3.348125, 3.450000, 3.553750, 3.660000,
                3.766250, 3.880000, 4.010625, 4.150000, 4.293125, 4.440000, 4.586250,
                4.740000, 4.908125, 5.090000, 5.288750, 5.500000, 5.720624, 5.950000,
        ])

let metalAlKSpectrum = PiecewiseLinearSpectrum(
        lambdas: metalWavelengths,
        values: [
                3.593750, 3.640000, 3.689375, 3.740000, 3.789375, 3.840000, 3.894375,
                3.950000, 4.005000, 4.060000, 4.113750, 4.170000, 4.233750, 4.300000,
                4.365000, 4.430000, 4.493750, 4.560000, 4.633750, 4.710000, 4.784375,
                4.860000, 4.938125, 5.020000, 5.108750, 5.200000, 5.290000, 5.380000,
                5.480000, 5.580000, 5.690000, 5.800000, 5.915000, 6.030000, 6.150000,
                6.280000, 6.420000, 6.550000, 6.700000, 6.850000, 7.000000, 7.150000,
                7.310000, 7.480000, 7.650000, 7.820000, 8.010000, 8.210000, 8.390000,
                8.570000, 8.620000, 8.600000, 8.450000, 8.310000, 8.210000, 8.210000,
        ])

// Copper extinction coefficients
let metalCuKSpectrum = PiecewiseLinearSpectrum(
        lambdas: metalWavelengths,
        values: [
                1.662125, 1.687000, 1.703313, 1.720000, 1.744563, 1.770000, 1.791625,
                1.810000, 1.822125, 1.834000, 1.851750, 1.872000, 1.894250, 1.916000,
                1.931688, 1.950000, 1.972438, 2.015000, 2.121562, 2.210000, 2.177188,
                2.130000, 2.160063, 2.210000, 2.249938, 2.289000, 2.326000, 2.362000,
                2.397625, 2.433000, 2.469187, 2.504000, 2.535875, 2.564000, 2.589625,
                2.605000, 2.595562, 2.583000, 2.576500, 2.599000, 2.678062, 2.809000,
                3.010750, 3.240000, 3.458187, 3.670000, 3.863125, 4.050000, 4.239563,
                4.430000, 4.619563, 4.817000, 5.034125, 5.260000, 5.485625, 5.717000,
        ])

var namedSpectra: [String: any Spectrum] = [
        "metal-Ag-eta": metalAgEtaSpectrum,
        "metal-Ag-k": metalAgKSpectrum,
        "metal-Al-eta": metalAlEtaSpectrum,
        "metal-Al-k": metalAlKSpectrum,
        "metal-Cu-eta": metalCuEtaSpectrum,
        "metal-Cu-k": metalCuKSpectrum,
]

public struct BaseRGBSpectrum<T: FloatingPoint>: Initializable, Three {

        init() {
                self.init(r: 0, g: 0, b: 0)
        }

        init(x: T, y: T, z: T) {
                self.x = x
                self.y = y
                self.z = z
        }

        init(r: T, g: T, b: T) {
                self.init(x: r, y: g, z: b)
        }

        init(intensity: T) {
                self.x = intensity
                self.y = intensity
                self.z = intensity
        }

        init(rgb: (T, T, T)) {
                self.x = rgb.0
                self.y = rgb.1
                self.z = rgb.2
        }

        var isNaN: Bool {
                return r.isNaN || g.isNaN || b.isNaN
        }

        var isInfinite: Bool {
                return r.isInfinite || g.isInfinite || b.isInfinite
        }

        // Conform to Three
        var x: T
        var y: T
        var z: T

        // Convenience accessors for RGBSpectrum
        var r: T {
                get { return x }
                set { x = newValue }
        }

        var g: T {
                get { return y }
                set { y = newValue }
        }

        var b: T {
                get { return z }
                set { z = newValue }
        }
}

extension BaseRGBSpectrum: CustomStringConvertible {

        public var description: String {
                return "[ \(r) \(g) \(b) ]"
        }
}

extension BaseRGBSpectrum {

        var isBlack: Bool {
                return r == 0 && g == 0 && b == 0
        }
}

extension BaseRGBSpectrum where T: FloatingPoint {

        public static func * (mul: T, spectrum: BaseRGBSpectrum<T>) -> BaseRGBSpectrum {
                return BaseRGBSpectrum(
                        r: mul * spectrum.x,
                        g: mul * spectrum.y,
                        b: mul * spectrum.z)
        }

        public static func * (spectrum: BaseRGBSpectrum<T>, mul: T) -> BaseRGBSpectrum {
                return mul * spectrum
        }

        public static func *= (left: inout BaseRGBSpectrum<T>, right: BaseRGBSpectrum<T>) {
                left.x *= right.x
                left.y *= right.y
                left.z *= right.z
        }

        public static func + (spectrum: BaseRGBSpectrum<T>, value: T) -> BaseRGBSpectrum {
                return BaseRGBSpectrum(
                        r: spectrum.r + value,
                        g: spectrum.g + value,
                        b: spectrum.b + value)
        }

        public static func - (spectrum: BaseRGBSpectrum<T>, value: T) -> BaseRGBSpectrum {
                return BaseRGBSpectrum(
                        r: spectrum.r - value,
                        g: spectrum.g - value,
                        b: spectrum.b - value)
        }

        public static func / (
                numerator: BaseRGBSpectrum<T>,
                denominator: BaseRGBSpectrum<T>
        ) -> BaseRGBSpectrum {
                return BaseRGBSpectrum(
                        r: numerator.r / denominator.r,
                        g: numerator.g / denominator.g,
                        b: numerator.b / denominator.b)
        }

        public static func == (a: BaseRGBSpectrum<T>, b: BaseRGBSpectrum<T>) -> Bool {
                return a.r == b.r && a.g == b.g && a.b == b.b
        }

        public static func != (a: BaseRGBSpectrum<T>, b: BaseRGBSpectrum<T>) -> Bool {
                return !(a == b)
        }

        public func squareRoot() -> BaseRGBSpectrum {
                return BaseRGBSpectrum(
                        r: r.squareRoot(),
                        g: g.squareRoot(),
                        b: b.squareRoot())
        }

        func average() -> T {
                return (r + g + b) / 3
        }

        var maxValue: T {
                return max(r, max(g, b))
        }

        subscript(index: Int) -> T {
                get {
                        switch index {
                        case 0: return r
                        case 1: return g
                        case 2: return b
                        default: return r
                        }
                }
                set(newValue) {
                        switch index {
                        case 0: r = newValue
                        case 1: g = newValue
                        case 2: b = newValue
                        default: r = newValue
                        }
                }
        }

}

extension BaseRGBSpectrum where T: BinaryFloatingPoint {

        func asRgb() -> RGBSpectrum {
                return RGBSpectrum(r: FloatX(r), g: FloatX(g), b: FloatX(b))
        }

        var luminance: T {
                let rw: T = 0.212671 * r
                let gw: T = 0.715160 * g
                let bw: T = 0.072169 * b
                return rw + gw + bw
        }
}

extension BaseRGBSpectrum where T: FloatingPoint {

        init(from normal: Normal3<T>) {
                let normal = normalized(normal)
                self.init(r: abs(normal.x), g: abs(normal.y), b: abs(normal.z))
        }
}

typealias RGBSpectrum = BaseRGBSpectrum<FloatX>

extension RGBSpectrum: Spectrum {}

let black = RGBSpectrum(intensity: 0)
let gray = RGBSpectrum(intensity: 0.5)
let white = RGBSpectrum(intensity: 1)
let red = RGBSpectrum(r: 1, g: 0, b: 0)
let blue = RGBSpectrum(r: 0, g: 0, b: 1)
let green = RGBSpectrum(r: 0, g: 1, b: 0)

func pow(base: RGBSpectrum, exp: FloatX) -> RGBSpectrum {
        return RGBSpectrum(
                r: pow(base.r, exp),
                g: pow(base.g, exp),
                b: pow(base.b, exp))
}

func exp(_ x: RGBSpectrum) -> RGBSpectrum {
        return RGBSpectrum(
                r: exp(x.r),
                g: exp(x.g),
                b: exp(x.b))
}

func gammaLinearToSrgb(light: RGBSpectrum) -> RGBSpectrum {
        var converted = RGBSpectrum()
        converted.r = gammaLinearToSrgb(value: light.r)
        converted.g = gammaLinearToSrgb(value: light.g)
        converted.b = gammaLinearToSrgb(value: light.b)
        return converted
}

func gammaSrgbToLinear(light: RGBSpectrum) -> RGBSpectrum {
        var converted = RGBSpectrum()
        converted.r = gammaSrgbToLinear(value: light.r)
        converted.g = gammaSrgbToLinear(value: light.g)
        converted.b = gammaSrgbToLinear(value: light.b)
        return converted
}
