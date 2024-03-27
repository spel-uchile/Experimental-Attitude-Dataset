
import numpy

__all__ = ['generate_binary_structure']


def _center_is_true(structure, origin):
    structure = numpy.array(structure)
    coor = tuple([oo + ss // 2 for ss, oo in zip(structure.shape,
                                                 origin)])
    return bool(structure[coor])


def generate_binary_structure(rank, connectivity):
    if connectivity < 1:
        connectivity = 1
    if rank < 1:
        return numpy.array(True, dtype=bool)
    output = numpy.fabs(numpy.indices([3] * rank) - 1)
    output = numpy.add.reduce(output, 0)
    return output <= connectivity
