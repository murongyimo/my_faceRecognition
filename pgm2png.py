from PIL import Image  
import os.path
import glob 

INPUT_DATA_DIR = './dataset/orl_faces'
OUTPUT_DATA_DIR = './dataset/orl_faces_png'

#将pgm格式转化为png格式，以便提供给tensorflow进行处理

def pgm2png(in_dir, out_dir):  
    if not os.path.exists(out_dir):  
        print(out_dir, 'is not existed.')  
        os.mkdir(out_dir)  
    if not os.path.exists(in_dir):  
        print(in_dir, 'is not existed.')  
        return -1  

    for sub_dir in glob.glob(in_dir+'/*'):  
        #sub_dir:各个子文件夹
        print("processing:", sub_dir)
        #out_sub_dir:转存的各个子文件夹,文件从sub_dir中读取，存储到out_sub_dir中
        out_sub_dir=os.path.join(out_dir,os.path.basename(sub_dir))
        print("out_sub_dir:",out_sub_dir)
        #创建相应的文件夹
        if not os.path.exists(out_sub_dir):  
            print(out_sub_dir, 'is not existed.')  
            os.mkdir(out_sub_dir)
        for files in glob.glob(sub_dir+'/*.'+'pgm'):  
            #读取每个文件夹，得到路径、文件名、后缀，进行分割，从而组合得到输出文件相对路径
            filepath, filename = os.path.split(files) 
            outfile, _ = os.path.splitext(filename)
            outfile = outfile+'.png'
            #out:./dataset/orl_faces/s32 ----- 5.pgm ------> 5.jpg
            print(filepath,'-----',filename,'------>',outfile)
            img = Image.open(files) 
            new_path = os.path.join(out_sub_dir, outfile) 
            img.save(new_path) 
        print("--------------") 

if __name__=='__main__':  
    pgm2png(INPUT_DATA_DIR, OUTPUT_DATA_DIR)  
