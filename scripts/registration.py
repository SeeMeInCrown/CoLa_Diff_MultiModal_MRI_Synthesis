import os
# import fsl
t1_image_path='/home/cmet-standard/Downloads/IXI-T1'
t2_image_path='/home/cmet-standard/Downloads/IXI-T2'
pd_image_path='/home/cmet-standard/Downloads/IXI-PD'
t1_names=os.listdir(t1_image_path)
t1_names.sort()
t2_names=os.listdir(t2_image_path)
t2_names.sort()
pd_names=os.listdir(pd_image_path)
pd_names.sort()
t1_image = [t1_image_path + '/' + t1_name for t1_name in t1_names]
t2_image = [t2_image_path + '/' + t2_name for t2_name in t2_names]
pd_image = [pd_image_path + '/' + pd_name for pd_name in pd_names]
t1_std = [t1_image_path + '/stdT1' + t1_name for t1_name in t1_names]
t2_std = [t2_image_path + '/stdT2' + t2_name for t2_name in t2_names]
pd_std=[pd_image_path + '/stdPD' + pd_name for pd_name in pd_names]
t1_reg=[t1_image_path + '/regT1/' +t1_name for t1_name in t1_names]
t2_reg=[t2_image_path + '/regT2/' +t2_name for t2_name in t2_names]
pd_reg=[pd_image_path + '/regPD/' +pd_name for pd_name in pd_names]
# print(t1_image)
# print(t2_image)
# print(pd_image)
count=0

#for pd
for i in range(len(t2_names)):
    if (os.path.exists(pd_image_path + '/' + t2_names[i].split('-')[0] + '-' + t2_names[i].split('-')[1] + '-' +
                       t2_names[i].split('-')[2] + '-PD.nii.gz')):
        count = count + 1
        # (t2_names[i].split('-')[0:3]==t1_names[i].split('-')[0:3])&(t2_names[i].split('-')[0:3]==pd_names[i].split('-')[0:3])):
        rest_pd = pd_image_path + '/' + t2_names[i].split('-')[0] + '-' + t2_names[i].split('-')[1] + '-' + \
                  t2_names[i].split('-')[2] + '-PD.nii.gz'
        print(rest_pd)
        # print(count)
        cmd1 = 'fslreorient2std' + ' ' + t1_image[i]
        # cmd='flirt -in '+t2_image[i]+' -ref '+rest_t1[i]+' -out '+t2_reg[i]+' -omat '+'m2f.mat -dof 6'
        cmd = 'flirt -in ' + rest_pd + ' -ref ' + t2_image[i] + ' -out ' + pd_reg[i] + ' -omat ' + 'm2f.mat -dof 6'
        os.system(cmd)
