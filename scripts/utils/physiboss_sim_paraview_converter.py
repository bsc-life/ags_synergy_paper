import os
import glob
import scipy.io
import xml.etree.ElementTree as ET
from vtk import (vtkUnstructuredGrid, vtkPoints, vtkVertex, vtkDoubleArray,
                vtkCellArray, vtkXMLUnstructuredGridWriter)

def parse_physiboss_labels(xml_file):
    """Parse the XML file to get labels and metadata for each field"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Find the labels section in the XML
    labels_elem = root.find(".//labels")
    if labels_elem is None:
        raise ValueError("Could not find labels in XML file")
    
    # Parse each label
    labels = {}
    for label in labels_elem.findall('label'):
        index = int(label.get('index'))
        size = int(label.get('size'))
        units = label.get('units')
        name = label.text.strip()
        labels[index] = {'name': name, 'size': size, 'units': units}
    
    return labels

def mat_to_vtu(mat_file, xml_file, output_file):
    """Convert a PhysiCell .mat file to VTU format with proper labels"""
    try:
        # Get labels from XML
        labels = parse_physiboss_labels(xml_file)
        
        # Load mat file
        data = scipy.io.loadmat(mat_file)
        cells = data['cells']
        
        # Debug print
        print(f"Processing {os.path.basename(mat_file)}")
        print(f"Cells array shape: {cells.shape}")
        
        # Create VTK grid
        grid = vtkUnstructuredGrid()
        points = vtkPoints()
        
        # Get number of cells
        num_cells = cells.shape[1]
        
        # Add points to the grid (positions are at indices 1,2,3)
        for i in range(num_cells):
            x = float(cells[1][i])
            y = float(cells[2][i])
            z = float(cells[3][i])
            points.InsertNextPoint(x, y, z)
        
        grid.SetPoints(points)
        
        # Create cells (vertices)
        vertices = vtkCellArray()
        for i in range(num_cells):
            vertex = vtkVertex()
            vertex.GetPointIds().SetId(0, i)
            vertices.InsertNextCell(vertex)
        grid.SetCells(vtkVertex().GetCellType(), vertices)
        
        # Add cell data
        for idx, label_info in labels.items():
            name = label_info['name']
            size = label_info['size']
            
            # Skip position data as it's already handled
            if name == "position":
                continue
            
            try:
                # Create array for this field
                array = vtkDoubleArray()
                array.SetName(name)
                
                if size == 1:
                    # Scalar data
                    array.SetNumberOfComponents(1)
                    for i in range(num_cells):
                        value = float(cells[idx][i])
                        array.InsertNextValue(value)
                else:
                    # Vector data
                    array.SetNumberOfComponents(size)
                    for i in range(num_cells):
                        vector = [float(cells[idx + j][i]) for j in range(size)]
                        array.InsertNextTuple(vector)
                
                grid.GetCellData().AddArray(array)
            except Exception as e:
                print(f"Warning: Skipping field {name} due to error: {str(e)}")
                continue
        
        # Write to file
        writer = vtkXMLUnstructuredGridWriter()
        writer.SetFileName(output_file)
        writer.SetInputData(grid)
        writer.Write()
        
        return True
        
    except Exception as e:
        print(f"Error processing file {os.path.basename(mat_file)}: {str(e)}")
        return False

def create_pvd(output_dir, time_interval=40):
    """Create a ParaView Data (PVD) file indexing all timesteps"""
    # Find all mat files
    mat_files = sorted(glob.glob(os.path.join(output_dir, "*_cells.mat")))
    if not mat_files:
        raise ValueError(f"No .mat files found in {output_dir}")
    
    # Create output directory for VTU files
    vtu_dir = os.path.join(output_dir, "vtu")
    os.makedirs(vtu_dir, exist_ok=True)
    
    # PVD file header
    pvd_content = ['<?xml version="1.0"?>',
                   '<VTKFile type="Collection" version="0.1">',
                   '  <Collection>']
    
    successful_conversions = 0
    
    # Process each timestep
    for i, mat_file in enumerate(mat_files):
        # Get corresponding XML file
        xml_file = mat_file.replace("_cells.mat", ".xml")
        if not os.path.exists(xml_file):
            print(f"Warning: XML file not found for {mat_file}")
            continue
        
        # Create VTU filename
        vtu_file = os.path.join(vtu_dir, f"cells_{i:08d}.vtu")
        
        print(f"\nProcessing timestep {i}...")
        
        # Convert mat to vtu
        if mat_to_vtu(mat_file, xml_file, vtu_file):
            # Add to PVD index only if conversion was successful
            timestamp = i * time_interval
            relative_path = os.path.relpath(vtu_file, output_dir)
            pvd_content.append(f'    <DataSet timestep="{timestamp}" group="" part="0" file="{relative_path}"/>')
            successful_conversions += 1
    
    # Close PVD file
    pvd_content.extend(['  </Collection>',
                       '</VTKFile>'])
    
    # Write PVD file only if we have successful conversions
    if successful_conversions > 0:
        pvd_file = os.path.join(output_dir, "cells.pvd")
        with open(pvd_file, 'w') as f:
            f.write('\n'.join(pvd_content))
        print(f"\nSuccessfully converted {successful_conversions} timesteps")
    else:
        print("\nNo successful conversions")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python physiboss_sim_paraview_converter.py <output_directory>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    create_pvd(output_dir)