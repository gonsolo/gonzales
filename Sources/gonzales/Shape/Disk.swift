final class Disk: Shape {

        init(objectToWorld: Transform, radius: FloatX) {
                self.objectToWorld = objectToWorld
                self.radius = radius
        }

        @MainActor
        func worldBound() -> Bounds3f {
                return objectToWorld * objectBound()
        }

        @MainActor
        func objectBound() -> Bounds3f {
                // TODO
                return Bounds3f()
        }

        func intersect(
                ray worldRay: Ray,
                tHit: inout FloatX,
                interaction: inout SurfaceInteraction
        ) throws {
                // TODO
        }

        func area() -> FloatX {
                fatalError("Not implemented")
        }

        func sample(u: TwoRandomVariables) -> (interaction: any Interaction, pdf: FloatX) {
                // TODO
                return (SurfaceInteraction(), 0)
        }

        public var description: String {
                return "Disk"
        }

        static func statistics() {
                print("TODO")
        }

        let objectToWorld: Transform
        let radius: FloatX
}

func createDiskShape(objectToWorld: Transform, parameters: ParameterDictionary) throws -> [any Shape] {
        let radius = try parameters.findOneFloatX(called: "radius", else: 1.0)
        let disk = Disk(objectToWorld: objectToWorld, radius: radius)
        return [disk]
}
