import bpy
import os
from mathutils import Vector

# --- Configuration ---
# !!! SET THIS TO THE PATH OF YOUR .TXT SKELETON FILE !!!
skeleton_filepath = "/home/zrs/control_gnn/sample_sk/val/gs_7.5/1ef08b46cc8b4d7dbf28ecac87975934.txt" # <--- CHANGE THIS

armature_name = "ImportedSkeleton" # Name for the new armature object in Blender
bone_prefix = "Joint_" # Prefix for bone names in Blender
# --- End Configuration ---

def parse_skeleton_file(filepath):
    """Parses the custom skeleton .txt file."""
    joint_positions = {}
    hierarchy = []
    root_joint_index = None

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Skeleton file not found: {filepath}")

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if not parts:
                continue

            command = parts[0].lower()

            try:
                if command == "root" and len(parts) == 2:
                    joint_name = parts[1]
                    if joint_name.startswith("joint"):
                        root_joint_index = int(joint_name[len("joint"):])
                    else:
                        print(f"Warning: Unexpected root format: {line}")
                elif command == "joints" and len(parts) == 5:
                    joint_name = parts[1]
                    if joint_name.startswith("joint"):
                        joint_index = int(joint_name[len("joint"):])
                        x = float(parts[2])
                        y = float(parts[3])
                        z = float(parts[4])
                        joint_positions[joint_index] = Vector((x, y, z)) # Use Vector
                    else:
                        print(f"Warning: Unexpected joints format: {line}")
                elif command == "hier" and len(parts) == 3:
                    parent_name = parts[1]
                    child_name = parts[2]
                    if parent_name.startswith("joint") and child_name.startswith("joint"):
                        parent_index = int(parent_name[len("joint"):])
                        child_index = int(child_name[len("joint"):])
                        hierarchy.append((parent_index, child_index))
                    else:
                        print(f"Warning: Unexpected hier format: {line}")
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line: {line} - Error: {e}")

    if root_joint_index is None:
        print("Warning: Root joint not found in file.")
    if not joint_positions:
        raise ValueError("No joint positions found in file.")
    if not hierarchy:
        raise ValueError("No hierarchy information found in file.")

    return joint_positions, hierarchy, root_joint_index

def create_armature_from_data(name, joint_positions, hierarchy, root_joint_index):
    """Creates a Blender armature from parsed skeleton data."""

    # Create a new armature object and data
    bpy.ops.object.add(type='ARMATURE', enter_editmode=True, align='WORLD', location=(0, 0, 0))
    armature_obj = bpy.context.object
    armature_obj.name = name
    armature_data = armature_obj.data
    armature_data.name = name + "_Data"

    # Bones are created in Edit Mode
    edit_bones = armature_data.edit_bones

    # --- Remove the default bone created by armature_add ---
    # Check if there's a bone to remove first
    if len(edit_bones) > 0:
        default_bone = edit_bones[0]
        edit_bones.remove(default_bone)
        print("Removed default bone.")
    #--------------------------------------------------------

    print(f"Creating {len(hierarchy)} bones...")
    created_bones = {} # Store created bones: child_index -> bone

    # --- Create bones based on hierarchy ---
    # A bone connects a parent joint to a child joint
    for parent_idx, child_idx in hierarchy:
        if parent_idx not in joint_positions or child_idx not in joint_positions:
            print(f"Warning: Missing position data for hierarchy link ({parent_idx} -> {child_idx}). Skipping bone.")
            continue

        parent_pos = joint_positions[parent_idx]
        child_pos = joint_positions[child_idx]

        # Bone name often corresponds to the child joint it points to
        bone_name = f"{bone_prefix}{child_idx}"
        bone = edit_bones.new(name=bone_name)

        # Set bone head (start) and tail (end) positions
        bone.head = parent_pos
        bone.tail = child_pos

        # Store the created bone, keyed by the child index it points TO
        created_bones[child_idx] = bone
        print(f"  Created bone: {bone_name} from joint{parent_idx} to joint{child_idx}")

    # --- Set up parenting ---
    print("Setting up bone hierarchy...")
    for parent_idx, child_idx in hierarchy:
        if child_idx not in created_bones:
            # This bone creation failed earlier
            continue

        child_bone = created_bones[child_idx]

        # Find the parent bone. The parent bone is the one that *points to* the parent_idx.
        # So, we look for the bone stored with the key parent_idx in our created_bones dict.
        parent_bone = created_bones.get(parent_idx)

        if parent_bone:
            child_bone.parent = parent_bone
            # Optionally connect the child bone's head to the parent bone's tail
            child_bone.use_connect = True
            print(f"  Parented {child_bone.name} to {parent_bone.name}")
        else:
            # This happens if the parent_idx is the root or wasn't a child in any other hier line
            print(f"  Bone {child_bone.name} has no parent bone in the created set (possibly child of root).")


    # Switch back to Object Mode to see the result
    bpy.ops.object.mode_set(mode='OBJECT')
    print("Armature creation complete.")
    return armature_obj

# --- Main Execution ---
if __name__ == "__main__":
    try:
        print(f"Parsing skeleton file: {skeleton_filepath}")
        positions, hier, root_idx = parse_skeleton_file(skeleton_filepath)
        print(f"Found {len(positions)} joints and {len(hier)} hierarchy links.")
        if root_idx is not None:
            print(f"Root joint index: {root_idx}")

        create_armature_from_data(armature_name, positions, hier, root_idx)

        print("Script finished successfully.")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        # You might want Blender to show an error pop-up
        # bpy.ops.wm.call_menu(name="INFO_MT_error_popup") # This needs context handling
    except ValueError as e:
        print(f"ERROR in file format or content: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()