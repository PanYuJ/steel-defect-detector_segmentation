def tile_image(df, save_path, load_path, subset='train', tile_w_ratio=0.25, tile_h_ratio=1.0):

  """
  Arg:
    df: Dataframe contain image ID and mask rle
    save_path: The root of directory for saving cropped image and mask
    tile_w_ratio: The width ratio of slice image and mask
    tile_h_ratio: The height ratio of slice image and mask
  """
  
  # Check whether directory is exist in "save_path", or it create directory.
  if subset=='train':
    img_save_path = os.path.join(save_path, 'train_tile_images')
    
  if subset=='val':
    img_save_path = os.path.join(save_path, 'val_tile_images')
    
  if not os.path.isdir(img_save_path):
    os.makedirs(img_save_path)

  tile_df = pd.DataFrame()
  img_tile_list = []
  rle_tile_list = []
  
  for idx in tqdm(range(len(df))):
    # Load image from Dataframe
    img_root = os.path.join(load_path, df['ImageId'].iloc[idx])
    img = cv2.imread(img_root, cv2.IMREAD_GRAYSCALE)
    
    # Load mask from Dataframe
    mask = np.zeros(shape=(256,1600,4))
    for i in range(4):
      mask[:, :, i] = rle2mask(df['e'+str(i+1)].iloc[idx])
      
    # Define size of image and mask after cropping
    tile_w = img.shape[1]*tile_w_ratio
    tile_h = img.shape[0]*tile_h_ratio

    # Cropping and saving
    for w in range(int(img.shape[1]/tile_w)):
      for h in range(int(img.shape[0]/tile_h)):

        # Cropping image
        img_tile = img[int(h*crop_h):int((h+1)*tile_h),int(w*tile_w):int((w+1)*tile_w)]

        # Define image name
        image_name = df['ImageId'].iloc[idx]+'_{}-{}.jpg'.format(w,h)
        save_root_img = os.path.join(img_save_path, image_name)
        img_tile_list.append(image_name)
        cv2.imwrite(img=img_tile, filename=save_root_img)

        # Cropping mask
        mask_tile = mask[int(h*tile_h):int((h+1)*tile_h), int(w*tile_w):int((w+1)*tile_w), :]
        for i in range(4):
          if mask_tile[:,:,i].sum()!=0:
            rle = mask2rle(mask_tile[:,:,i])
          else:
            rle=''
          rle_tile_list.append(rle)
          

  tile_df['ImageId'] = img_tile_list      
  tile_df['e1'] = rle_tile_list[0::4]
  tile_df['e2'] = rle_tile_list[1::4]
  tile_df['e3'] = rle_tile_list[2::4]
  tile_df['e4'] = rle_tile_list[3::4]
  return tile_df 
