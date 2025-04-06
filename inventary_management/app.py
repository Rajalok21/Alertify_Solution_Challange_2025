import test_gps
import tsp


def memain():
    ##for storing coordinates
    lat1 = 22.488540958444293
    long1 = 88.36633469004033
    L1 = test_gps.read_coordinates_from_file('ndrf_small_units.txt',lat1, long1)


    ##applying tsp
    filename = 'adjacency_matrix.txt'
    path=tsp.solve_tsp(filename)
    print(path)

    #mapping

    test_gps.mapping(L1,path)
