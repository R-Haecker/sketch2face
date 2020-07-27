import os
import numpy as np
import PIL.Image as Image
import torch
import torchvision
import torchvision.transforms as transforms
import sys 
import yaml
import matplotlib.pyplot as plt
from skimage.transform import resize

cur_path = os.path.dirname(os.path.abspath(__file__))
cur_path = cur_path.replace("\\", "/")
path = cur_path[:cur_path.rfind('/')]
sys.path.append(path)
from model.wgan import CycleWGAN_GP_VAE
from data.data_loader.dataset import Dataset
'''cur_path = os.path.dirname(os.path.abspath(__file__))
path = cur_path[:cur_path.rfind('/')]
sys.path.append(path)'''

def croped_images_from_batch(batch_path, image_size=32, save_path=None, save_name=None, to_gray=True):
    if save_name is None:
        save_name = ''
    else:
        save_name = save_name + '_'
    image_type = batch_path[(batch_path.rfind('.')):]
    batch_img = np.asarray(Image.open(batch_path).convert('RGB'))
    batch_shape = batch_img.shape
    images = []
    for i in range(batch_shape[0]//image_size):
        for j in range(batch_shape[1]//image_size):
            images.append(batch_img[image_size*i: image_size*(i+1), image_size*j: image_size*(j+1), :])
    images = np.array(images)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        for i,image in enumerate(images):
            pil_image = Image.fromarray(image)
            pil_image.save(os.path.join(save_path, save_name + str(i)) + image_type)
    if to_gray:
        images = rgb2gray(images)
    return torch.tensor(images).float()

def rgb2gray(rgb):
    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def get_transformation(size=32):
    transformation = transforms.Compose([transforms.ToPILImage(), transforms.Resize(size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])  
    return transformation

def restore(model, config, checkpoint_path):
    state = torch.load(checkpoint_path, map_location='cpu')
    model.netG_A.enc.load_state_dict(state['sketch_encoder'])
    model.netG_B.dec.load_state_dict(state['sketch_decoder'])
    model.netD_A.load_state_dict(state['sketch_discriminator'])
    model.netG_B.enc.load_state_dict(state['face_encoder'])
    model.netG_A.dec.load_state_dict(state['face_decoder'])
    model.netD_B.load_state_dict(state['face_dicriminator'])
    if config['variational']['num_latent_layer'] > 0:
        model.netG_A.latent_layer.load_state_dict(state['sketch_latent_layer'])
        model.netG_B.latent_layer.load_state_dict(state['face_latent_layer'])

def load_model(config, checkpoint_dir, checkpoint_name):
    device='cpu'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    model = CycleWGAN_GP_VAE(config).to(device)
    restore(model, config, checkpoint_path)
    return model

def get_same_images(N, img=None, img_path=None, to_gray=True):
    assert (img is None and img_path is not None)  or (img is not None and img_path is None), "Either img or img_path must or must be specified."
    if img_path is not None:
        img = np.asarray(Image.open(img_path).convert('RGB'))
    images = np.array([img for i in range(N)])
    if to_gray:
        images = rgb2gray(images)
    return torch.tensor(images).float()

def get_config(config_path):
    with open(config_path) as file:
        config = yaml.full_load(file)
    return config

def unix_path(path):
    return path.replace("\\", "/")

def transform_batch(transformation, batch):
    new_batch = []
    for img in batch:
        new_batch.append(transformation(img))
    return torch.stack(new_batch).float()

#x = croped_images_from_batch("C:/Users/user/Desktop/Uni/Semester 8/Deep Vision/sketch2face/eval/data/2020-07-22T12-55-18_s2f_load_train_all/train/validation/log_op/batch_input_sketch_0018000.png",
#                            save_path="data/test_batch", save_name="first_test")
#transformation = get_transformation(size=32)
#x = transform_batch(transformation,x )

def save_predictions_same_image(img_num, name_add=''):
    checkpoint_dir = unix_path("C:\\Users\\user\\Desktop\\Uni\\Semester 8\\Deep Vision\\sketch2face\\eval\\data\\2020-07-22T12-55-18_s2f_load_train_all\\train\\checkpoints")
    checkpoint_name = "model-18192.ckpt"
    config_path = unix_path("C:\\Users\\user\\Desktop\\Uni\\Semester 8\\Deep Vision\\sketch2face\\config\\07_22\\cycle_wgan_continue_full.yaml")

    #img_num = 7
    img_path = "data/test_batch/first_test_{}.png".format(img_num)

    config = get_config(config_path)

    model = load_model(config, checkpoint_dir, checkpoint_name)

    transformation = get_transformation(size=32)

    batch = get_same_images(10, img_path=img_path)



    
    batch = transform_batch(transformation, batch)
    plt.imshow((batch[0][0].numpy() + 1)/2 )
    plt.savefig("data/test_batch_output/{}{}_input".format(img_num, name_add))

    with torch.no_grad():
        output = model(batch)

    output = output[0].numpy().transpose(0,2,3,1)
    for i,img in enumerate(output):
        plt.imshow((img+1)/2)
        plt.savefig("data/test_batch_output/{}{}_version_{}".format(img_num, name_add, i))

def plot_enc_output():
    checkpoint_dir = unix_path("C:\\Users\\user\\Desktop\\Uni\\Semester 8\\Deep Vision\\sketch2face\\eval\\data\\2020-07-22T12-55-18_s2f_load_train_all\\train\\checkpoints")
    checkpoint_name = "model-18192.ckpt"
    config_path = unix_path("C:\\Users\\user\\Desktop\\Uni\\Semester 8\\Deep Vision\\sketch2face\\config\\07_22\\cycle_wgan_continue_full.yaml")

    
    config = get_config(config_path)
    model = load_model(config, checkpoint_dir, checkpoint_name)

    transformation = get_transformation(size=32)

    z_std_list = []
    z2_std_list = []

    for name in ["0018000", "0017000", "0016000", "0015000"]:
        batch = croped_images_from_batch("C:/Users/user/Desktop/Uni/Semester 8/Deep Vision/sketch2face/eval/data/2020-07-22T12-55-18_s2f_load_train_all/train/validation/log_op/batch_input_sketch_{}.png".format(name))
        batch = transform_batch(transformation, batch)

        with torch.no_grad():
            output = model(batch)
        
        z = model.netG_A.z.numpy()
        z2 = model.netG_A.z2.numpy()

        #mu = z[:,:256]
        #sig = z[:,256:]

        #mu_mu = np.mean(mu, axis=0)
        z_std = np.mean(np.std(z, axis=0))
        z2_std = np.mean(np.std(z2, axis=0))
        print("z_std:", z_std)
        print("z2_std:", z2_std)
        
def save_predictions_same_image_dataloader(img_num, name_add=''):
    checkpoint_dir = unix_path("C:\\Users\\user\\Desktop\\Uni\\Semester 8\\Deep Vision\\sketch2face\\eval\\data\\2020-07-22T12-55-18_s2f_load_train_all\\train\\checkpoints")
    checkpoint_name = "model-18192.ckpt"
    config_path = unix_path("C:\\Users\\user\\Desktop\\Uni\\Semester 8\\Deep Vision\\sketch2face\\config\\07_22\\cycle_wgan_continue_full.yaml")

    
    config = get_config(config_path)
    model = load_model(config, checkpoint_dir, checkpoint_name)

    model.eval()

    dataloader = Dataset(config, train=False)

    img = dataloader[img_num]['image_sketch']

    batch = torch.stack([img for i in range(10)])

    plt.imshow((batch[0][0].numpy() + 1)/2 )
    plt.savefig("data/test_batch_output/{}{}_input".format(img_num, name_add))

    with torch.no_grad():
        output = model(batch)
    output = output[0].numpy().transpose(0,2,3,1)
    for i,img in enumerate(output):
        plt.imshow((img+1)/2)
        plt.savefig("data/test_batch_output/{}{}_version_{}".format(img_num, name_add, i))


def save_predictions_different_image_dataloader(img_num, name_add=''):
    checkpoint_dir = unix_path("C:\\Users\\user\\Desktop\\Uni\\Semester 8\\Deep Vision\\sketch2face\\eval\\data\\2020-07-22T12-55-18_s2f_load_train_all\\train\\checkpoints")
    checkpoint_name = "model-18192.ckpt"
    config_path = unix_path("C:\\Users\\user\\Desktop\\Uni\\Semester 8\\Deep Vision\\sketch2face\\config\\07_22\\cycle_wgan_continue_full.yaml")

    
    config = get_config(config_path)
    model = load_model(config, checkpoint_dir, checkpoint_name)

    model.eval()

    dataloader = Dataset(config, train=False)

    batch = torch.stack([dataloader[i]['image_sketch'] for i in range(img_num, img_num+10)])

    plt.imshow((batch[0][0].numpy() + 1)/2 )
    plt.savefig("data/test_batch_output/{}{}_input".format(img_num, name_add))

    with torch.no_grad():
        output = model(batch)
    output = output[0].numpy().transpose(0,2,3,1)
    for i,img in enumerate(output):
        plt.imshow((img+1)/2)
        plt.savefig("data/test_batch_output/{}{}_image_{}".format(img_num, name_add, i))

model_type='only_latent'

#x = croped_images_from_batch("C:/Users/user/Desktop/Uni/Semester 8/Deep Vision/sketch2face/eval/data/{}/input.png".format(model_type), image_size=32, save_path="C:/Users/user/Desktop/Uni/Semester 8/Deep Vision/sketch2face/eval/data/{}/input".format(model_type), save_name="")
#x = croped_images_from_batch("C:/Users/user/Desktop/Uni/Semester 8/Deep Vision/sketch2face/eval/data/{}/fake.png".format(model_type), image_size=64, save_path="C:/Users/user/Desktop/Uni/Semester 8/Deep Vision/sketch2face/eval/data/{}/fake".format(model_type), save_name="")
#x = croped_images_from_batch("C:/Users/user/Desktop/Uni/Semester 8/Deep Vision/sketch2face/eval/data/{}/rec.png".format(model_type), image_size=32, save_path="C:/Users/user/Desktop/Uni/Semester 8/Deep Vision/sketch2face/eval/data/{}/rec".format(model_type), save_name="")

#save_predictions_different_image_dataloader(7, "_3_different")
#plot_enc_output()
'''batch_path = None#"path/to/batch/image"
save_path = None#"path/to/dir/of/images"
save_name = None#"name"'''

def get_image_collection(model_type, indeces, columns=2, base_path ="C:/Users/user/Desktop/Uni/Semester 8/Deep Vision/sketch2face/eval/data/"):
    
    
    for image_type in ["input", "fake", "rec"]:
        path = unix_path(os.path.join(base_path, model_type, image_type))
        title = image_type + " sketch"
        if image_type == "fake":
            title = image_type + " face"
        if image_type == "rec":
            title = image_type + "onstruction sketch"
        images = []
        for i in indeces:
            image_path = unix_path(os.path.join(path, "_{}.png".format(i)))
            images.append(np.asarray(Image.open(image_path).convert('RGB')))
        images = np.array(images)
        image_sets = [images[columns*i : columns*(i+1)] for i in range(len(images)//columns)]
        image_sets = np.concatenate(image_sets, axis=-3)
        image_sets = np.concatenate(image_sets, axis=-2)
        image_sets = resize(image_sets, (128*len(images)//columns, 256), order=0)
        
        save_path = os.path.join(path, "collection")
        plt.title(title)
        plt.imshow(image_sets, interpolation='none')
        plt.axis('off')
        plt.savefig(save_path ,bbox_inches='tight')
        plt.show()

def save_predictions_same_image_dataloader2(N, img_num_list, model_type, config_path, checkpoint_dir, checkpoint_name, name_add='', offset=0):
    #checkpoint_dir = unix_path("C:\\Users\\user\\Desktop\\Uni\\Semester 8\\Deep Vision\\sketch2face\\eval\\data\\2020-07-22T12-55-18_s2f_load_train_all\\train\\checkpoints")
    #checkpoint_name = "model-18192.ckpt"
    #config_path = unix_path("C:\\Users\\user\\Desktop\\Uni\\Semester 8\\Deep Vision\\sketch2face\\config\\07_22\\cycle_wgan_continue_full.yaml")

    
    config = get_config(config_path)
    for img_num in img_num_list:
        model = load_model(config, checkpoint_dir, checkpoint_name)

        model.eval()

        dataloader = Dataset(config, train=False)

        img = dataloader[img_num]['image_sketch']

        batch = torch.stack([img for i in range(offset, offset+N)])


        save_path = "data/{}/{}{}_".format(model_type, img_num, name_add)
        plt.title('input sketch')
        plt.imshow(resize((batch[0][0].numpy() + 1)/2 , (128, 128), order=0),cmap='gray')
        plt.axis('off')
        
        plt.savefig(save_path + "input", bbox_inches='tight')
        plt.show()

        with torch.no_grad():
            output = model(batch)
        #print(type(output))
        #print(np.(output[0].numpy().transpose(0,2,3,1), axis=-2).shape)
        output =  resize((np.concatenate(output[0].numpy().transpose(0,2,3,1), axis=-2 )+1)/2, (128, 128*N, 3), order=0)
        rec = resize((np.concatenate(model.output["rec_A"].numpy().transpose(0,2,3,1)[...,0], axis=-1)+1)/2, (128, 128*N), order=0)

        plt.title("fake face")
        plt.imshow(output, interpolation='none')
        plt.axis('off')
        plt.savefig(save_path + "fake" ,bbox_inches='tight')
        plt.show()

        plt.title("reconstruction sketch")
        plt.imshow(rec, interpolation='none', cmap='gray')
        plt.axis('off')
        plt.savefig(save_path + "rec" ,bbox_inches='tight')
        plt.show()


        
        
        
        
save_predictions_same_image_dataloader2(3, [23, 24, 20, 11, 13], 'from_skratch',
        config_path = unix_path("C:\\Users\\user\\Desktop\\Uni\\Semester 8\\Deep Vision\\sketch2face\\config\\07_21\\cycle_wgan_from_skratch.yaml"),
        checkpoint_dir = unix_path("C:\\Users\\user\\Desktop\\Uni\\Semester 8\\Deep Vision\\sketch2face\\eval\\data\\sketch2face_run_data\\from_scratch"),
        checkpoint_name = "model-100000.ckpt")


#for name in ["only_latent","train_all"]:
#    get_image_collection(name, [1,2,3,7,5,6])


