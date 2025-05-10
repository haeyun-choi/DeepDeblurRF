import os
import subprocess

####################################################################
########     COLMAP WRAPPER FOR DEEPDEBLURRF (CVPR 2025)     #######
####################################################################

def run_colmap(basedir, match_type):
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')

    feature_extractor_args = ['xvfb-run', '-a',
        'colmap', 'feature_extractor',
        '--database_path', os.path.join(basedir, 'database.db'),
        '--image_path', os.path.join(basedir, 'images'),
        '--ImageReader.camera_model', 'SIMPLE_PINHOLE',
        '--ImageReader.single_camera', '1',
        # The following two lines were removed because COLMAP does not support them
        # '--ImageReader.estimate_affine_shape', '1',
        # '--ImageReader.domain_size_pooling', '1',
    ]

    feat_output = subprocess.check_output(feature_extractor_args, universal_newlines=True)
    logfile.write(feat_output)
    print('[COLMAP] Features extracted.')

    # --- Feature matching with guided matching ---
    exhaustive_matcher_args = ['xvfb-run', '-a',
        'colmap', match_type,
        '--database_path', os.path.join(basedir, 'database.db'),
        '--SiftMatching.guided_matching', '1',
    ]
    match_output = subprocess.check_output(exhaustive_matcher_args, universal_newlines=True)
    logfile.write(match_output)
    print('[COLMAP] Features matched.')

    # --- Create sparse directory if not exists ---
    sparse_path = os.path.join(basedir, 'sparse')
    if not os.path.exists(sparse_path):
        os.makedirs(sparse_path)

    # --- Mapping with COLAMP-specific settings ---
    mapper_args = [
        'colmap', 'mapper',
        '--database_path', os.path.join(basedir, 'database.db'),
        '--image_path', os.path.join(basedir, 'images'),
        '--output_path', sparse_path,
        '--Mapper.min_num_matches', '16',
        '--Mapper.tri_min_angle', '1.5',
        '--Mapper.num_threads', '16',
        '--Mapper.init_min_tri_angle', '4',
        '--Mapper.multiple_models', '0',
        '--Mapper.extract_colors', '0',
    ]
    map_output = subprocess.check_output(mapper_args, universal_newlines=True)
    logfile.write(map_output)
    logfile.close()
    print('[COLMAP] Sparse map created.')
    print(f'[COLMAP] Finished, see {logfile_name} for logs.')