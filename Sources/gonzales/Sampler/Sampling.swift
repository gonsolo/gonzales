import Foundation

func concentricSampleDisk(u: TwoRandomVariables) -> Point2f {
        let u = Point2f(x: u.0, y: u.1)
        let uOffset = 2.0 * u - Vector2F(x: 1, y: 1)
        if uOffset.x == 0 && uOffset.y == 0 {
                return Point2f()
        }
        var theta: FloatX = 0.0
        var r: FloatX = 0.0
        if abs(uOffset.x) > abs(uOffset.y) {
                r = uOffset.x
                theta = (FloatX.pi / 4.0) * (uOffset.y / uOffset.x)
        } else {
                r = uOffset.y
                theta = (FloatX.pi / 2.0) - (FloatX.pi / 4.0) * (uOffset.x / uOffset.y)
        }
        return r * Point2f(x: cos(theta), y: sin(theta))
}

func cosineSampleHemisphere(u: TwoRandomVariables) -> Vector {
        let d = concentricSampleDisk(u: u)
        let z = sqrt(max(0, 1 - d.x * d.x - d.y * d.y))
        return Vector(x: d.x, y: d.y, z: z)
}

func powerHeuristic(f: FloatX, g: FloatX) -> FloatX {
        if f == 0 || g == 0 { return 0 }
        return (f * f) / (f * f + g * g)
}
