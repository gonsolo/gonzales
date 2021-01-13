import Foundation

final class TransformedPrimitive: Boundable & Intersectable {

        init(primitive: Boundable & Intersectable, transform: Transform) {
                self.primitive = primitive
                self.transform = transform
        }

        func intersect(ray: Ray, tHit: inout FloatX) throws -> SurfaceInteraction? {
                let localRay = transform.inverse * ray
                // TODO: transform tHit?
                guard var intersection = try primitive.intersect(ray: localRay, tHit: &tHit) else { return nil }
                intersection = transform * intersection
                return intersection
        }

        func worldBound() -> Bounds3f {
                let bound = transform * primitive.worldBound()
                return bound
        }

        func objectBound() -> Bounds3f {
                unimplemented()
        }

        let primitive: Boundable & Intersectable
        let transform: Transform
}

