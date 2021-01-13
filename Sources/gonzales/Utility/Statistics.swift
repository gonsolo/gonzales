struct Statistics {

        func report() {
                print("Statistics:")
                renderer?.camera.statistics()
                scene.statistics()
                Triangle<Int>.statistics()
                //Curve.statistics()
                BoundingHierarchyBuilder.statistics()
                //Bounds3.statistics()
        }
}

var statistics = Statistics()

