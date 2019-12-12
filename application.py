from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
from train import *


def predict(image_path, model, class_names):
    img = cv2.imread(image_path)
    img = torch.from_numpy(img / 255).float().permute(2, 0, 1)
    img = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img) 
    img = torch.tensor(img).unsqueeze(0).to(device, dtype=torch.float)
    outputs = model(img)
    _, preds = torch.max(outputs, 1)
    # print(preds)
    preds = preds.numpy()[0]
    return class_names[preds]

def interface(model, class_names):
    root = Tk()
    root.title('Applications')
    lbl = Label(root, text='Image')
    image_p = ''
    lbl.grid(column=0, row=0)
    def clicked():
        root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        print (root.filename)
        img = Image.open(root.filename)
        img = img.resize((250, 250), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(img)
        img_open = Label(root, image=image)
        img_open.image = image
        img_open.place(x=0, y=30)

        logo_pred = predict(root.filename, model, class_names)

        logo_label = Label(root, font=("Arial Bold", 15))
        prediction = Label(root, text='Predicted:')
        prediction.place(x=300, y=130)
        logo_label.configure(text=logo_pred)
        logo_label.place(x = 300, y=150)

    btn_open = Button(root, text='Open', command=clicked)
    btn_open.grid(column=2, row=0)
    root.geometry('500x500')
    root.mainloop()
    return root.filename

if __name__ == "__main__":
    #Load model
    model = model_init()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load('checkpoint.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    data_loader, dataset_sizes, class_names = data_loader()
    img = interface(model, class_names)
    print(img)