main_path = '/Volumes/solo/ali/'


config = {'stage1_data_path':main_path + 'stage1/stage1/',
          'luna_raw':main_path + 'luna/raw/',
          'luna_segment':main_path + 'luna/seg-lungs-LUNA16/',
          
          'luna_data':main_path + 'luna/allset/',
          'preprocess_result_path':main_path + 'stage1/preprocess/',
          
          'luna_abbr':main_path + 'luna/detector/labels/shorter.csv',
          'luna_label':main_path + 'luna/detector/labels/lunaqualified.csv',
          'stage1_annos_path':[main_path + 'luna/detector/labels/label_job5.csv',
                main_path + 'luna/detector/labels/label_job4_2.csv',
                main_path + 'luna/detector/labels/label_job4_1.csv',
                main_path + 'luna/detector/labels/label_job0.csv',
                main_path + 'luna/detector/labels/label_qualified.csv'],
          'bbox_path':main_path + 'luna/detector/results/res18/bbox/',
          'preprocessing_backend':'python'
         }

