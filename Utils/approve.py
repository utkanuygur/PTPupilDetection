import os
import shutil
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import random

class ImageSelector:
    def __init__(self, train_dir, output_dir):
        self.train_dir = train_dir
        self.output_dir = output_dir
        self.applied_images = self.get_applied_images()
        self.current_batch = 0
        self.selected_images = set()
        
        self.fig, self.axes = plt.subplots(3, 3, figsize=(15, 15))
        plt.subplots_adjust(bottom=0.1)
        self.setup_gui()

    def get_applied_images(self):
        applied_dir = os.path.join(self.train_dir, 'applied')
        images_dir = os.path.join(self.train_dir, 'images')
        masks_dir = os.path.join(self.train_dir, 'masks')
        
        applied_images = []
        for f in os.listdir(applied_dir):
            if f.endswith('.jpg'):
                image_path = os.path.join(images_dir, f)
                mask_path = os.path.join(masks_dir, f.replace('.jpg', '.gif'))
                if os.path.exists(image_path) and os.path.exists(mask_path):
                    applied_images.append(os.path.join(applied_dir, f))
        
        random.shuffle(applied_images)  # Shuffle the order of images
        return applied_images

    def setup_gui(self):
        self.next_button_ax = plt.axes([0.81, 0.05, 0.1, 0.04])
        self.next_button = Button(self.next_button_ax, 'Next Batch')
        self.next_button.on_clicked(self.next_batch)

        self.finish_button_ax = plt.axes([0.7, 0.05, 0.1, 0.04])
        self.finish_button = Button(self.finish_button_ax, 'Finish')
        self.finish_button.on_clicked(self.finish)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def show_batch(self):
        start = self.current_batch * 9
        batch = self.applied_images[start:start+9]
        
        for ax in self.axes.ravel():
            ax.clear()
            ax.axis('off')

        for ax, img_path in zip(self.axes.ravel(), batch):
            if img_path:
                img = plt.imread(img_path)
                ax.imshow(img)
                ax.set_title(os.path.basename(img_path), fontsize=8)
                if img_path in self.selected_images:
                    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='red', lw=3, transform=ax.transAxes))

        self.fig.canvas.draw()

    def on_click(self, event):
        if event.inaxes in self.axes.ravel():
            img_index = list(self.axes.ravel()).index(event.inaxes)
            img_path = self.applied_images[self.current_batch * 9 + img_index]
            
            if img_path in self.selected_images:
                self.selected_images.remove(img_path)
            else:
                self.selected_images.add(img_path)
            
            self.show_batch()

    def on_key_press(self, event):
        if event.key == ' ':
            self.save_selected_images()
            print("Progress saved.")

    def next_batch(self, event):
        self.save_selected_images()
        self.current_batch += 1
        if self.current_batch * 9 >= len(self.applied_images):
            self.current_batch = 0
        self.show_batch()
        print("Progress saved and moved to next batch.")

    def finish(self, event):
        self.save_selected_images()
        plt.close()
        print("Finished. All selected images have been saved.")

    def save_selected_images(self):
        for subdir in ['images', 'masks', 'applied']:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)

        for img_path in self.selected_images:
            img_name = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(self.output_dir, 'applied', img_name))
            shutil.copy(os.path.join(self.train_dir, 'images', img_name), 
                        os.path.join(self.output_dir, 'images', img_name))
            shutil.copy(os.path.join(self.train_dir, 'masks', img_name.replace('.jpg', '.gif')), 
                        os.path.join(self.output_dir, 'masks', img_name.replace('.jpg', '.gif')))

def main():
    train_directory = "train"
    output_directory = "train_approved"
    
    selector = ImageSelector(train_directory, output_directory)
    selector.show_batch()
    plt.show()

if __name__ == "__main__":
    main()
