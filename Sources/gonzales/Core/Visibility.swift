/**
        A type that can calculate the visibilty from a point to another.
*/
struct Visibility {

        func unoccluded() throws -> Bool {
                var (ray, tHit) = from.spawnRay(to: to.position)
                return try scene.intersect(ray: ray, tHit: &tHit) == nil
        }

        var from: Interaction
        var to: Interaction
}

