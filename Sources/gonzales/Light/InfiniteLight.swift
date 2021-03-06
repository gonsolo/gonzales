import Foundation // sin, cos

struct InfiniteLight: Light {

        init(lightToWorld: Transform, brightness: Spectrum, texture: Texture<Spectrum>) {
                self.lightToWorld = lightToWorld
                self.brightness = brightness
                self.texture = texture
        }

        func sample(for reference: Interaction, u: Point2F) -> (radiance: Spectrum,
                                                                direction: Vector,
                                                                pdf: FloatX,
                                                                visibility: Visibility) {
                let theta = u[1] * FloatX.pi
                let phi = u[0] * 2 * FloatX.pi
                let lightDirection = Vector(x: sin(theta) * cos(phi),
                                              y: sin(theta) * sin(phi),
                                              z: cos(theta))
                let direction = lightToWorld * lightDirection
                let pdf = theta < machineEpsilon ? 0 : 1 / (2 * FloatX.pi * FloatX.pi * sin(theta))
                let distantPoint = SurfaceInteraction(position: reference.position + direction * scene.diameter())
                let visibility = Visibility(from: reference, to: distantPoint)
		let uv = directionToUV(direction: -direction)
                let interaction = SurfaceInteraction(uv: uv)
		let color = texture.evaluate(at: interaction)
                return (radiance: color, direction, pdf, visibility)
        }

        func probabilityDensityFor(samplingDirection direction: Vector, from reference: Interaction) throws -> FloatX {
                let incoming = worldToLight * direction
                let (theta, _) = sphericalCoordinatesFrom(vector: incoming)
                guard theta > machineEpsilon else { return 0 }
                let pdf: FloatX = 1 / (2 * FloatX.pi * FloatX.pi * sin(theta))
                return pdf
        }

        let inv2Pi: FloatX = 1.0 / (2.0 * FloatX.pi)

        private func sphericalPhi(_ v: Vector) -> FloatX {
                let p = atan2(v.y, v.x)
                if p < 0.0 {
                        return p + 2.0 * FloatX.pi
                } else {
                        return p
                }
        }

        private func sphericalTheta(_ v: Vector) -> FloatX {
                return acos(clamp(value: v.z, low: -1, high: 1))
        }

        private func directionToUV(direction: Vector) -> Point2F {
                let w = normalized(worldToLight * direction)
                let u = sphericalPhi(w) * inv2Pi
                var v = sphericalTheta(w) * inv2Pi
                v = 2 * (1 - v) // PBRT compatibility. Where does this come from?
                let uv = Point2F(x: u, y: v)
                return uv
        }

        func radianceFromInfinity(for ray: Ray) -> Spectrum {
                let uv = directionToUV(direction: ray.direction)
                let interaction = SurfaceInteraction(uv: uv)
                let radiance = texture.evaluate(at: interaction)
                return radiance
        }

        var isDelta: Bool { get { return false } }

        var worldToLight: Transform { get { return lightToWorld.inverse } }

        let brightness: Spectrum
        let texture: Texture<Spectrum>
        let lightToWorld: Transform
}

func createInfiniteLight(lightToWorld: Transform, parameters: ParameterDictionary) throws -> InfiniteLight {
        guard let mapname = try parameters.findString(called: "mapname") else {
                let brightness = try parameters.findSpectrum(name: "L") ?? white
                let texture = ConstantTexture(value: brightness)
                return InfiniteLight(lightToWorld: lightToWorld, brightness: brightness, texture: texture)
        }
        let texture = try getTextureFrom(name: mapname)
        return InfiniteLight(lightToWorld: lightToWorld, brightness: white, texture: texture)
}

