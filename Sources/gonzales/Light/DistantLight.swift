import Foundation

struct DistantLight {

        init(lightToWorld: Transform, brightness: RgbSpectrum, direction: Vector) {
                self.brightness = brightness
                self.direction = normalized(lightToWorld * direction)
        }

        func sample(for reference: Interaction, u: TwoRandomVariables) -> (
                radiance: RgbSpectrum, direction: Vector, pdf: FloatX, visibility: Visibility
        ) {
                let outside = reference.position + direction * 2 * worldRadius
                let visibility = Visibility(
                        from: reference, to: SurfaceInteraction(position: outside))
                return (brightness, direction, pdf: 1, visibility)
        }

        func probabilityDensityFor(samplingDirection direction: Vector, from reference: Interaction)
                throws -> FloatX
        {
                return 0
        }

        func radianceFromInfinity(for ray: Ray) -> RgbSpectrum { return black }

        func power() -> FloatX {
                let value = square(worldRadius) * FloatX.pi * brightness.average()
                return value
        }

        var isDelta: Bool { return true }

        let direction: Vector
        let brightness: RgbSpectrum
        let worldRadius: FloatX = 100.0  // TODO
}

func createDistantLight(lightToWorld: Transform, parameters: ParameterDictionary) throws
        -> DistantLight
{
        let from = try parameters.findOnePoint(name: "from", else: origin)
        let to = try parameters.findOnePoint(name: "to", else: origin)
        guard let brightness = try parameters.findSpectrum(name: "L") as? RgbSpectrum else {
                throw ParameterError.missing(parameter: "L", function: #function)
        }
        let direction: Vector = from - to
        return DistantLight(
                lightToWorld: lightToWorld, brightness: brightness, direction: direction)
}
