/**
        A basic geometric shape like a triangle or a sphere.
        It encapsulates the basic ingredients of a shape in that it can be
        transformed, bounded, intersected and sampled.
*/
protocol Shape: AnyObject, Transformable, Boundable, Intersectable, Sampleable {}

