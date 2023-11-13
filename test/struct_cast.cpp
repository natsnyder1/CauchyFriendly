#include <stdio.h>
#include <stdlib.h>

struct Point3D
{
    double x;
    double y;
    double z;
};

int main()
{
    const int N = 12;
    Point3D* points = (Point3D*) malloc( N * sizeof(Point3D) );
    for(int i = 0; i < N; i++)
    {
        points[i].x = 3*i;
        points[i].y = 3*i + 1;
        points[i].z = 3*i + 2;
    }

    double* points_as_array = (double*) points;
    for(int i = 0; i < 3*N; i++)
        printf("%.1lf, ", points_as_array[i]);
    printf("\n");
    free(points);
    return 0;
}