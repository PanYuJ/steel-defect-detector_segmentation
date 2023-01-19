def crop_image(df, save_path, load_path, subset='train', crop_w_ratio=0.25, crop_h_ratio=1.0):

  """
  Arg:
    df: Dataframe contain image ID and mask rle
    save_path: The root of directory for saving cropped image and mask
    crop_w_ratio: The width ratio of slice image and mask
    crop_h_ratio: The height ratio of slice image and mask
  """
  
  # Check whether directory is exist in "save_path", or it create directory.
  if subset=='train':
    img_save_path = os.path.join(save_path, 'train_crop_images')
    
  if subset=='val':
    img_save_path = os.path.join(save_path, 'val_crop_images')
    
  if not os.path.isdir(img_save_path):
    os.makedirs(img_save_path)

  crop_df = pd.DataFrame()
  img_crop_list = []
  rle_crop_list = []
  
  for idx in tqdm(range(len(df))):
    # Load image from Dataframe
    img_root = os.path.join(load_path, df['ImageId'].iloc[idx])
    img = cv2.imread(img_root, cv2.IMREAD_GRAYSCALE)
    
    # Load mask from Dataframe
    mask = np.zeros(shape=(256,1600,4))
    for i in range(4):
      mask[:, :, i] = rle2mask(df['e'+str(i+1)].iloc[idx])
      
    # Define size of image and mask after cropping
    crop_w = img.shape[1]*crop_w_ratio
    crop_h = img.shape[0]*crop_h_ratio

    # Cropping and saving
    for w in range(int(img.shape[1]/crop_w)):
      for h in range(int(img.shape[0]/crop_h)):

        # Cropping image
        img_crop = img[int(h*crop_h):int((h+1)*crop_h),int(w*crop_w):int((w+1)*crop_w)]

        # Define image name
        image_name = df['ImageId'].iloc[idx]+'_{}-{}.jpg'.format(w,h)
        save_root_img = os.path.join(img_save_path, image_name)
        img_crop_list.append(image_name)
        cv2.imwrite(img=img_crop, filename=save_root_img)

        # Cropping mask
        mask_crop = mask[int(h*crop_h):int((h+1)*crop_h), int(w*crop_w):int((w+1)*crop_w), :]
        for i in range(4):
          if mask_crop[:,:,i].sum()!=0:
            rle = mask2rle(mask_crop[:,:,i])
          else:
            rle=''
          rle_crop_list.append(rle)
          

  crop_df['ImageId'] = img_crop_list      
  crop_df['e1'] = rle_crop_list[0::4]
  crop_df['e2'] = rle_crop_list[1::4]
  crop_df['e3'] = rle_crop_list[2::4]
  crop_df['e4'] = rle_crop_list[3::4]
  return crop_df 
