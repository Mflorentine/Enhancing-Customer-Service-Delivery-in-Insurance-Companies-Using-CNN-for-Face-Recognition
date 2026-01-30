#  # Default Camera
#                 video_capture = cv2.VideoCapture(0)
#                 # detector = MTCNN()
        
#                 while True:
#                     # capture frame by frame
#                     (grabbed,frame)= video_capture.read()
#                     # ret, img_cam = video_capture.read()
#                     rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#                     # output = detector.detect_faces(img_cam)
                    
#                     # Detect faces in the webcam
#                     faces = face_cascade.detectMultiScale(rgb,scaleFactor =1.3, minNeighbors =5)
                    
#                     # for each face found
                    
#                     for (x,y,w,h) in faces:
#                         roi_rgb = rgb[y:y+h,x:x+w]
                    
#                         # Draw a rectangle
#                         color = (255,0,0)#in BGR
#                         stroke=2
#                         cv2.rectangle(frame,(x,y),(x+w,y+h),color,stroke)
                        
#                         # Resizing the image
#                         size = (image_width,image_height)
#                         resized_image = cv2.resize(roi_rgb,size)
#                         image_array = np.array(resized_image,"uint8")
#                         img = image_array.reshape(1,image_width,image_height,3)
#                         img = img.astype('float32')
#                         img /= 255
                    
                        
                    
                    
#                     # for single_output in output:
#                     #     x,y,width,height = single_output["box"]
#                     #     cv2.rectangle(img_cam,pt1=(x,y),pt2=(x+width,y+height),color=(255,0,0),thickness=3)
                        
#                     # cv2.imshow("win",img_cam )
#                     # save the image in a directory
#                         if save_uploaded_image(img):
#                             # load the image
#                             display_image = Image.open(img)

#                             # extract the features
#                             features = extract_features(os.path.join(uploadn_path,img.name),model,detector)
#                             # recommend
#                             index_pos = recommend(feature_list,features)
#                             predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
#                             # display
#                             st.write("Accessing Highest Similarity Cosine")
#                             st.write(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][1])
                            
#                             similarity_big_index = (sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][1])*100
#                             threshold_similarity = 60
                            
#                             if similarity_big_index >= threshold_similarity:
                            
#                                 col1,col2 = st.columns(2)

#                                 with col1:
#                                     st.header('Your uploaded image')
#                                     st.image(display_image)
#                                 with col2:
#                                     st.header("Seems like " + predicted_actor)
#                                     st.image(filenames[index_pos],width=300)
                                    
#                             else:
#                                 st.write("UNKNOWN PERSON") 
#                                 col3,col4 = st.columns(2)

#                                 with col3:
#                                     st.header('Your uploaded image')
#                                     st.image(display_image)
#                                 with col4:
#                                     st.title("UNKNOWN")
#                                     # st.header("Seems like " + predicted_actor)
#                                     # st.image(filenames[index_pos],width=300)               
                            
                            
                            
#                         key = cv2.waitKey(1) & 0xFf
#                         if cv2.waitKey(1)==ord("q"):
#                         # if cv2.waitKey(1)==13:
#                             break
#                 video_capture.release()
#                 cv2.destroyAllWindows() 
# 





# from the app.py

            # =======================================================================================================================
                
            #     cap = cv2.VideoCapture(0)
            #     img_id = 0 
            #     while True:
            #         ret, frame = cap.read() # read the image from cap (camera)
            #         if face_cropped(frame) is not None: # Here frame is an argument img(the same)
                        
            #             # passing the image to be detected and once it is not None
            #             img_id +=1    
            #             # Resizing the image(frame)
            #             face = cv2.resize(face_cropped(frame), (64,64))
            #             # converting it to grayscale 
            #             #face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            #             # Locating the path to be stored later
            #             file_name_path = "artifacts/uploads/cropped/user."+str(img_id)+".jpg"
            #             # Saving the image in a folder
            #             cv2.imwrite(file_name_path, face)
            #             display_image = Image.open(file_name_path)
            #             # display_image.tile =[e for e in display_image.tile if e[1][2]<2181 and e[1][3]<1294]
            #             # Put(write some text) the text in my cropped image
            #             cv2.putText(face, str(img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            #             cv2.imshow("Cropped face", face)
            #             detector = MTCNN()
            #             # actors = os.listdir('C:/condascripts/faceMatch/artifacts/uploads/cropped')
            #             results = imgpath(file_name_path,detector)
            #             # for actor in actors:
            #                 # results = imgpath(actor,detector)
            #             if results:
            #                 features = extract_features(file_name_path,model,detector)
            #                 # st.write(features)
            #                 # recommend
            #                 index_pos = recommend(feature_list,features)
            #                 predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
            #                 # display
            #                 st.write("look at Here")
            #                 st.write(index_pos)
            #                 st.write(predicted_actor)
            #                 st.write("Accessing Highest Similarity Cosine")
            #                 st.write(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][1])

            #                 similarity_big_index = (sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][1])*100
            #                 threshold_similarity = 60
                            
            #                 if similarity_big_index >= threshold_similarity:
                            
            #                     col1,col2 = st.columns(2)

            #                     with col1:
            #                         st.header('Your uploaded image')
            #                         st.image(display_image)
            #                     with col2:
            #                         st.header("Seems like " + predicted_actor)
            #                         st.image(filenames[index_pos],width=300)
                                    
            #                     st.write(predicted_actor)
            #                     # st.write(predicted_actor)
                                
            #                     func_db1(predicted_actor)
            #                     func_db2(predicted_actor)
            #                     func_db3(predicted_actor)  
                            
            #                 else:
            #                     st.write("UNKNOWN PERSON") 
            #                     col3,col4 = st.columns(2)

            #                     with col3:
            #                         st.header('Your uploaded image')
            #                         st.image(display_image)
            #                     with col4:
            #                         st.title("UNKNOWN")                 
                    
            #         if cv2.waitKey(1)==13 or int(img_id)==5: #13 is the ASCII character of Enter key(break once enter key is pressed)
            #             # break it when enter key is pressed or when img_id (number of images is = 200)=200
            #             break
            #     # actors = os.listdir('C:/condascripts/faceMatch/artifacts/uploads/cropped')
            #     # for actor in actors: 
            #     #     os.remove(actor)                 
            # #       release my camera and destroy all windows       
            #     cap.release()
            #     cv2.destroyAllWindows()
            # ============================================================================================
            # if st.button("Detect Face"):
                
            #     cap = cv2.VideoCapture(0)
            #     img_id = 0 
            #     while True:
            #         ret, frame = cap.read() # read the image from cap (camera)
            #         if ret==True:
            #             if face_cropped(frame) is not None: # Here frame is an argument img(the same)
                        
            #                 # passing the image to be detected and once it is not None
            #                 img_id +=1    
            #                 # Resizing the image(frame)
            #                 face = cv2.resize(face_cropped(frame), (224,224))
            #                 # converting it to grayscale 
            #                 # face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            #                 # Locating the path to be stored later
            #                 # file_name_path ="artifacts/uploads/cropped/user."+str(img_id)+".jpg"
            #                 os.makedirs(f"artifacts/uploads/cropped/user{str(img_id)}")
            #                 file_name_path =f"artifacts/uploads/cropped/user{str(img_id)}/user."+str(img_id)+".jpg"
                            
            #                 # Saving the image in a folder
            #                 cv2.imwrite(file_name_path, face)
            #                 display_image = Image.open(file_name_path)
            #                 display_image.tile =[e for e in display_image.tile if e[1][2]<2181 and e[1][3]<1294]
            #                 # Put(write some text) the text in my cropped image
            #                 cv2.putText(face, str(img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            #                 cv2.imshow("Cropped face", face)
                            # detector = MTCNN()
            #                 # actors = os.listdir("artifacts/uploads/cropped/")

            #                 paths=os.listdir(f"C:/ACEDS docs/Face recognition/face-rwanda/artifacts/uploads/cropped/user{img_id}")
            #                 for actor in paths:
            #                     final_path = os.path.join(f"C:/ACEDS docs/Face recognition/face-rwanda/artifacts/uploads/cropped/user{img_id}",actor)
            #                     results = imgpath(final_path,detector)
                                
            #                 # paths=os.listdir(f"artifacts/uploads/cropped/user{img_id}")    results = imgpath(os.path.join("artifacts/uploads/cropped/",actor),detector)
            #                     if results:
            #                         # features = extract_features(os.path.join("artifacts/uploads/cropped/",actor),model,detector)
            #                         features = extract_features(final_path,model,detector)
            #                         # st.write(features)
            #                         # recommend
            #                         index_pos = recommend(feature_list,features)
            #                         predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
            #                         # display
            #                         st.write("look at Here")
            #                         st.write(index_pos)
            #                         st.write(predicted_actor)
            #                         st.write("Accessing Highest Similarity Cosine")
            #                         st.write(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][1])

            #                         similarity_big_index = (sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][1])*100
            #                         threshold_similarity = 55
                                    
            #                         if similarity_big_index >= threshold_similarity:
                                    
            #                             col1,col2 = st.columns(2)

            #                             with col1:
            #                                 st.header('Your uploaded image')
            #                                 st.image(display_image)
            #                             with col2:
            #                                 st.header("This face matches " + predicted_actor)
            #                                 st.image(filenames[index_pos],width=300)
                                            
            #                             st.write(predicted_actor)
            #                             # st.write(predicted_actor)
                                        
            #                             func_db1(predicted_actor)
            #                             func_db2(predicted_actor)
            #                             func_db3(predicted_actor)  
                                    
            #                         else:
            #                             st.write("UNKNOWN PERSON") 
            #                             col3,col4 = st.columns(2)

            #                             with col3:
            #                                 st.header('Your uploaded image')
            #                                 st.image(display_image)
            #                             with col4:
            #                                 st.title("UNKNOWN")                 
                            
            #         # if cv2.waitKey(1)==13 or int(img_id)==5: #13 is the ASCII character of Enter key(break once enter key is pressed)
            #         if cv2.waitKey(1)==13:     
                            # break it when enter key is pressed or when img_id (number of images is = 200)=200
                        # break
        # actors = os.listdir('C:/condascripts/faceMatch/artifacts/uploads/cropped')
        # for actor in actors: 
        #     os.remove(actor) 



            # if st.button("Detect Face"):
            #     detector = MTCNN()
            #     cap = cv2.VideoCapture(0)
            #     img_id=0
            #     while True:
            #         ret, frame = cap.read() # read the image from cap (camera)
            #         if ret==True:
            #             img_id +=1
                        
            #             face = face_cropped(frame) # Here frame is an argument img(the same)
            #             face = cv2.resize(face_cropped(frame), (200,200))
            #             # passing the image to be detected and once it is not None
            #             os.makedirs(f"C:/ACEDS docs/Face recognition/face-rwanda/artifacts/uploads/cropped/user{img_id}/")
            #             file_name_path =f"C:/ACEDS docs/Face recognition/face-rwanda/artifacts/uploads/cropped/user{img_id}/"+"user."+str(img_id)+".jpg"
            #             cv2.imwrite(file_name_path, face)
            #             paths=os.listdir(f"C:/ACEDS docs/Face recognition/face-rwanda/artifacts/uploads/cropped/user{img_id}")
            #             for actor in paths:
            #                 final_path = os.path.join(f"C:/ACEDS docs/Face recognition/face-rwanda/artifacts/uploads/cropped/user{img_id}/",actor)        
            #                 features = extract_features1(final_path,model,detector)

            #     #                 # recommend
            #                 index_pos = recommend(feature_list,features)
            #                 predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
            #             # display
            #                 st.write("look at Here")
            #                 st.write(index_pos)
            #                 st.write(predicted_actor)
            #                 st.write("Accessing Highest Similarity Cosine")
            #                 st.write(sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][1])

            #                 similarity_big_index = (sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][1])*100
            #                 threshold_similarity = 55
                            
            #                 if similarity_big_index >= threshold_similarity:
                            
            #                     col1,col2 = st.columns(2)

            #                     with col1:
            #                         st.header('Your uploaded image')
            #                         st.image(display_image)
            #                     with col2:
            #                         st.header("This face matches " + predicted_actor)
            #                         st.image(filenames[index_pos],width=300)
                                    
            #                     st.write(predicted_actor)
            #                     # st.write(predicted_actor)
                                
            #                     func_db1(predicted_actor)
            #                     func_db2(predicted_actor)
            #                     func_db3(predicted_actor)  
                            
            #                 else:
            #                     st.write("UNKNOWN PERSON") 
            #                     col3,col4 = st.columns(2)

            #                     with col3:
            #                         st.header('Your uploaded image')
            #                         st.image(display_image)
            #                     with col4:
            #                         st.title("UNKNOWN")                 
                    
            #         # if cv2.waitKey(1)==13 or int(img_id)==5: #13 is the ASCII character of Enter key(break once enter key is pressed)
            #         if cv2.waitKey(1)==13:     
            #                 # break it when enter key is pressed or when img_id (number of images is = 200)=200
            #             break            #    


                
  
                                    
            # #       release my camera and destroy all windows       
            #     cap.release()
            #     cv2.destroyAllWindows()      



                    