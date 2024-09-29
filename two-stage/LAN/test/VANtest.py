from model import VisualAttentionNetwork,myvan
from data import dataset_DALE
from train import train_utils
import torch
from torch.utils.data import DataLoader

def main():
    test_data_root = r'E:\ye\lol_dataset\our485\low'
    model_root = '../checkpoint/'
    test_result_root_dir = r'../testdata_van/LOL_VANguan/'

    # model_LessNet = LessNet.LessNet(stride=1)
    VisualAttentioNNet = VisualAttentionNetwork.VisualAttentionNetwork()  # LessNet_Update.LessNet()#AttentionNet.AttenteionNet(stride=1)
    state_dict = torch.load(model_root + 'VAN.pth')
    VisualAttentioNNet.load_state_dict(state_dict)

    test_data = dataset_DALE.DALETest(test_data_root)
    loader_test = DataLoader(test_data, batch_size=1, shuffle=False)

    VisualAttentioNNet.cuda()
    terst(loader_test, VisualAttentioNNet, test_result_root_dir)

def terst(loader_test, visualAttentionNet, root_dir) :
    visualAttentionNet.eval()
    for itr, data in enumerate(loader_test):
        testImg, fileName = data[0], data[1]
        testImg = testImg.cuda()
        print(itr,fileName)
        with torch.no_grad():
            test_attention_result= visualAttentionNet(testImg)

            test_recon_result_img = train_utils.tensor2im(test_attention_result)
            norm_input_img = train_utils.tensor2im(testImg+test_attention_result)

            recon_save_dir = root_dir + fileName[0]
            # recon_save_dir2 = root_dir + 'sum_'+fileName[0].split('.')[0]+('.png')

            train_utils.save_images(test_recon_result_img, recon_save_dir)
            # train_utils.save_images(norm_input_img, recon_save_dir)
            print(1)

if __name__ == '__main__':
    main()