import Foundation

struct Node {
        var bounds = Bounds3f()
        var count = 0
        var offset: Int = 0
        var axis: Int = 0
}

final class BoundingHierarchy: Boundable, Intersectable {

        init(primitives: [AnyObject & Intersectable], nodes: [Node]) {
                self.primitives = primitives
                self.nodes = nodes
        }

        func intersect(ray: Ray, tHit: inout FloatX) throws -> SurfaceInteraction? {
                if nodes.isEmpty { return nil }
                var interaction: SurfaceInteraction? = nil
                var toVisit = 0
                var current = 0
                var nodesToVisit = FixedArray64<Int>()
                
                while true {
                        let node = nodes[current]
                        if node.bounds.intersects(ray: ray, tHit: &tHit) {
                                if node.count > 0 {
                                        for i in 0..<node.count {
                                                let primitive = primitives[node.offset + i]
                                                interaction = try primitive.intersect(ray: ray, tHit: &tHit) ?? interaction
                                        }
                                        if toVisit == 0 { break }
                                        toVisit -= 1
                                        current = nodesToVisit[toVisit]
                                } else {
                                        if ray.direction[Int(node.axis)] < 0 {
                                                nodesToVisit[toVisit] = current + 1
                                                current = node.offset
                                        } else {
                                                nodesToVisit[toVisit] = node.offset
                                                current = current + 1
                                      }
                                      toVisit += 1
                                }
                        } else {
                                if toVisit == 0 { break }
                                toVisit -= 1
                                current = nodesToVisit[toVisit]
                        }
                }
                return interaction
        }

        func objectBound() -> Bounds3f {
                return worldBound()
        }

        func worldBound() -> Bounds3f {
                if nodes.isEmpty        { return Bounds3f() }
                else                    { return nodes[0].bounds }
        }

        var primitives: [AnyObject & Intersectable]
        var nodes: [Node]
}

