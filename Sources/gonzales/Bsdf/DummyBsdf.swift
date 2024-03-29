struct DummyBsdf: GlobalBsdf {

        func worldToLocal(world: Vector) -> Vector {
                return nullVector
        }

        func localToWorld(local: Vector) -> Vector {
                return nullVector
        }

        func albedo() -> RgbSpectrum { return white }

        func isReflecting(wi: Vector, wo: Vector) -> Bool { return false }

        func evaluateLocal(wo: Vector, wi: Vector) -> RgbSpectrum { return white }

        var bsdfFrame: BsdfFrame { return BsdfFrame() }
}
