/// A type that can calculate the visibility from a point to another.

struct Visibility {

        func unoccluded(scene: Scene) throws -> Bool {
                var (ray, tHit) = from.spawnRay(to: to.position)
                var interaction = SurfaceInteraction()
                try scene.intersect(
                        ray: ray,
                        tHit: &tHit,
                        interaction: &interaction)
                if interaction.valid {
                        return false
                } else {
                        return true
                }
        }

        let from: any Interaction
        let to: any Interaction
}
