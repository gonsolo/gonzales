func sameHemisphere(_ v1: Vector, _ v2: Vector) -> Bool { return v1.z * v2.z > 0 }

func absCosTheta(_ v: Vector) -> FloatX { return abs(v.z) }

func reflect(vector: Vector, by normal: Vector) -> Vector {
        return -vector + 2 * dot(vector, normal) * normal
}

func refract(wi: Vector, normal: Normal, eta: FloatX) -> Vector? {
        var wt = up
        let cosThetaI = dot(normal, wi)
        let sin2ThetaI = max(0, 1 - cosThetaI * cosThetaI)
        let sin2ThetaT = eta * eta * sin2ThetaI
        if sin2ThetaT >= 1 {
                return nil
        } else {
                let cosThetaT = (1 - sin2ThetaT).squareRoot()
                wt = eta * -wi + (eta * cosThetaI - cosThetaT) * Vector(normal: normal)
        }
        return wt
}

func sinTheta (_ w: Vector) -> FloatX { return sin2Theta(w).squareRoot() }
func sin2Theta(_ w: Vector) -> FloatX { return max(0, 1 - cos2Theta(w)) }
func cosTheta (_ w: Vector) -> FloatX { return w.z }
func cos2Theta(_ w: Vector) -> FloatX { return w.z * w.z }
func tan2Theta(_ w: Vector) -> FloatX { return sin2Theta(w) / cos2Theta(w) }
func tanTheta (_ w: Vector) -> FloatX { return sinTheta(w) / cosTheta(w) }

func sinPhi(_ w: Vector) -> FloatX {
        if sinTheta(w) == 0 {
                return 0
        } else {
                 return clamp(value: w.y / sinTheta(w), low: -1, high: 1);
        }
}

func sin2Phi(_ w: Vector) -> FloatX { return sinPhi(w) * sinPhi(w) }
func cos2Phi(_ w: Vector) -> FloatX { return cosPhi(w) * cosPhi(w) }

func cosPhi(_ w: Vector) -> FloatX {
        if sinTheta(w) == 0 {
                return 1
        } else {
                return clamp(value: w.x / sinTheta(w), low: -1, high: 1);
        }
}

