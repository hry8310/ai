import numpy as np
import cv2
import os

base_path='/root/mt/face2/'

#gen img list to .dat file
pos_dat_name='pos_list.dat'
neg_dat_name='neg_list.dat'

#gen the sample file'
vec_path=base_path+"pos.vec"

#init  img dir
pos_img_init_dir='pos2'
neg_img_init_dir='neg2'
# fit  img size to thi dir
pos_img_res_dir='pos'
neg_img_res_dir='neg'

#img size
width=50
height=50
###################
#custom config end
##################



pos_img_path=base_path+pos_img_init_dir
pos_dat_path=base_path+pos_dat_name
pos_out_path=base_path+pos_img_res_dir


neg_img_path=base_path+neg_img_init_dir
neg_dat_path=base_path+neg_dat_name
neg_out_path=base_path+neg_img_res_dir


def _resize(img_path,dat_path,out_path,dir,wh_dat=True):
    dat_file=open(dat_path,'w')
    cnt=0;
    for file in os.listdir(img_path):
        src = cv2.imread(os.path.join(img_path,file))
        src = cv2.resize(src, (width,height))
        of=os.path.join(out_path,file)
        cv2.imwrite(of,src)
        file_dir_info='./'+dir+'/'+file
        if wh_dat==True :
            file_dir_info =file_dir_info + ' 1 0 0 '+str(width)+" "+str(height) 
        file_dir_info=file_dir_info+'\n'
        dat_file.writelines( file_dir_info)
        cnt=cnt+1
    dat_file.close()
    return cnt

def gen_sample_sh( base_path,dat_name,vec_path ,cnt):
    print(vec_path)
    _str = " cd "+base_path+"; \n opencv_createsamples -info "+dat_name +"  -vec  " +vec_path+ "    -num "+str(cnt) +" -w " +str(width) + " -h "+str(height)
    _str = _str +';\n cd -'
    print(_str)
    return _str

def save_sample_sh( base_path,dat_name,vec_path ,cnt ,file='sample.sh', end_cmd=''):
    f=open(file,'w')
    _str=gen_sample_sh(base_path,dat_name,vec_path ,cnt)
    f.writelines(_str)
    f.writelines(end_cmd)
    f.close()
    os.system('chmod 777 ' +file)

def gen_train_sh(base_path , dat_name,vec_path,cade_dir='cascade' ,\
         param='  -numPos 100 -numNeg 50  -numStage 15 -w 50 -h 50  -mem 1024 -eqw 1 -mode ALL -bt GAB -minpos 32 '):
    _str= " cd "+base_path+"; \n opencv_traincascade -data "+cade_dir +" -vec "+vec_path+" -bg "+dat_name+param
    _str = _str +';\n cd -'
    print(_str)
    return _str

def save_train_sh(base_path , dat_name,vec_path,cade_dir='cascade' , file='train.sh' , end_cmd='', \
         param='  -numPos 100 -numNeg 50  -numStage 15 -w 50 -h 50  -mem 1024 -eqw 1 -mode ALL -bt GAB -minpos 32 '):
    f=open(file,'w')
    _str=gen_train_sh(base_path,dat_name,vec_path )
    f.writelines(_str)
    f.writelines(end_cmd)
    f.close()
    os.system('chmod 777 ' +file)

pos_cnt=_resize( pos_img_path,pos_dat_path,pos_out_path,pos_img_res_dir)
neg_cnt=_resize( neg_img_path,neg_dat_path,neg_out_path,neg_img_res_dir,wh_dat=False)

save_sample_sh(base_path,pos_dat_name,vec_path,pos_cnt)
save_train_sh(base_path,neg_dat_name,vec_path)
