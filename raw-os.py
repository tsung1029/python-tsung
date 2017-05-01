#!/usr/bin/env python
import sys
import h5py

RAW_TEMPLATE = """
<Xdmf>
    <Domain Name="Data">
        <Grid Name="electrons" GridType="Uniform">
            <Topology TopologyType="Polyvertex" NodesPerElement="{1}" />

            <Geometry GeometryType="X_Y_Z">

            <DataItem ItemType="Function" Dimensions="{1}" Function="$0 - {2}">
                    <DataItem DataType="Float" Precision="4" Dimensions="{1}" Format="HDF">
                        {0}:/x1
                    </DataItem>
                </DataItem>

                <DataItem DataType="Float" Precision="4" Dimensions="{1}" Format="HDF">
                    {0}:/x2
                </DataItem>

                <DataItem DataType="Float" Precision="4" Dimensions="{1}" Format="HDF">
                    {0}:/x3
                </DataItem>
            </Geometry>

            <Attribute AttributeType="Scalar" Center="Node" Name="ene">
                <DataItem DataType="Float" Precision="4" Dimensions="{1}" Format="HDF">
                    {0}:/p3
                </DataItem>
            </Attribute>

        </Grid>
    </Domain>
</Xdmf>
"""


if __name__ == "__main__":
    print sys.argv
    if len(sys.argv) < 2 :
        print "please add filename and number of particles, example: python raw.py RAW-electrons_N_5_7-000100.h5 "
        exit(0)

    rawfile = sys.argv[1]
    h5rawfile = h5py.File(rawfile)

    #get length
    particles = h5rawfile.get("p1").len()
    time_val = h5rawfile.attrs["TIME"][0]

    
    output = RAW_TEMPLATE.format(rawfile, particles, time_val)

    print "writing raw xmf file"
    outfilename = rawfile
    outfilename = outfilename.replace('.h5','.xmf')
    f = file(outfilename, "w")
    f.write(output)
    f.close()

