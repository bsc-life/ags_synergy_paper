# Othmane Hayoun-Mya 
# 2025-04-04

# This script is used to convert the MAT and XML files from a PhysiBoSS simulation 
# into a paraview readable format, .vtu and .pvd files.
# For now, only works for simple geometries (agents as spheres), and only for the cells, not the microenvironment..

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
    # Get labels from XML
    labels = parse_physiboss_labels(xml_file)
    
    # Load mat file
    data = scipy.io.loadmat(mat_file)
    cells = data['cells']
    
    # Create VTK grid
    grid = vtkUnstructuredGrid()
    
    # Create points from position data
    points = vtkPoints()
    positions = cells[1].reshape(-1, 3) if len(cells[1].shape) == 1 else cells[1]
    
    # Add points to the grid
    for i in range(len(positions)):
        points.InsertNextPoint(float(positions[i,0]), float(positions[i,1]), float(positions[i,2]))
    grid.SetPoints(points)
    
    # Create cells (vertices)
    vertices = vtkCellArray()
    for i in range(points.GetNumberOfPoints()):
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
            
        # Create array for this field
        array = vtkDoubleArray()
        array.SetName(name)
        
        # Handle scalar vs vector data
        if size == 1:
            array.SetNumberOfComponents(1)
            for value in cells[idx]:
                array.InsertNextValue(float(value))
        else:
            array.SetNumberOfComponents(size)
            data_reshaped = cells[idx].reshape(-1, size)
            for row in data_reshaped:
                array.InsertNextTuple(row.astype(float))
        
        grid.GetCellData().AddArray(array)
    
    # Write to file
    writer = vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(grid)
    writer.Write()

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
        
        print(f"Processing timestep {i}...")
        
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

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert PhysiBoSS output to Paraview format')
    parser.add_argument('output_dir', help='Directory containing PhysiBoSS output files')
    parser.add_argument('--time-interval', type=int, default=40, help='Time interval between frames (default: 40)')
    
    args = parser.parse_args()
    
    create_pvd(args.output_dir, args.time_interval)

if __name__ == '__main__':
    main()






