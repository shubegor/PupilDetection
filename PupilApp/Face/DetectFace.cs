using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
#if !(__IOS__ || NETFX_CORE)
using Emgu.CV.Cuda;
#endif

namespace PupilApp.Face
{
    public class DetectFace
    {


        public static void Detect(
           IInputArray image, 
           String faceFileName, 
           String eyeFileName,
           List<Rectangle> faces, 
           List<Rectangle> eyes, 
           out long detectionTime)
        {
            Stopwatch watch;

            using (InputArray iaImage = image.GetInputArray())
            {

            #if !(__IOS__ || NETFX_CORE)
                if (iaImage.Kind == InputArray.Type.CudaGpuMat && CudaInvoke.HasCuda)
                {
                    using (CudaCascadeClassifier face = new CudaCascadeClassifier(faceFileName))
                    using (CudaCascadeClassifier eye = new CudaCascadeClassifier(eyeFileName))
                    {
                        face.ScaleFactor = 1.1;
                        face.MinNeighbors = 10;
                        face.MinObjectSize = Size.Empty;
                        eye.ScaleFactor = 1.1;
                        eye.MinNeighbors = 10;
                        eye.MinObjectSize = Size.Empty;
                        watch = Stopwatch.StartNew();
                        using (CudaImage<Bgr, Byte> gpuImage = new CudaImage<Bgr, byte>(image))
                        using (CudaImage<Gray, Byte> gpuGray = gpuImage.Convert<Gray, Byte>())
                        using (GpuMat region = new GpuMat())
                        {
                            face.DetectMultiScale(gpuGray, region);
                            Rectangle[] faceRegion = face.Convert(region);
                            faces.AddRange(faceRegion);
                            foreach (Rectangle f in faceRegion)
                            {
                                using (CudaImage<Gray, Byte> faceImg = gpuGray.GetSubRect(f))
                                {
                                    //For some reason a clone is required.
                                    //Might be a bug of CudaCascadeClassifier in opencv
                                    using (CudaImage<Gray, Byte> clone = faceImg.Clone(null))
                                    using (GpuMat eyeRegionMat = new GpuMat())
                                    {
                                        eye.DetectMultiScale(clone, eyeRegionMat);
                                        Rectangle[] eyeRegion = eye.Convert(eyeRegionMat);
                                        foreach (Rectangle e in eyeRegion)
                                        {
                                            Rectangle eyeRect = e;
                                            eyeRect.Offset(f.X, f.Y);
                                            eyes.Add(eyeRect);
                                        }
                                    }
                                }
                            }
                        }
                        watch.Stop();
                    }
                }
                else
                #endif
                {
                    //Read the HaarCascade objects
                    using (CascadeClassifier face = new CascadeClassifier(faceFileName))
                    using (CascadeClassifier eye = new CascadeClassifier(eyeFileName))
                    {
                        watch = Stopwatch.StartNew();

                        using (UMat ugray = new UMat())
                        {
                            CvInvoke.CvtColor(image, ugray, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);

                            //normalizes brightness and increases contrast of the image
                            CvInvoke.EqualizeHist(ugray, ugray);

                            //Detect the faces  from the gray scale image and store the locations as rectangle
                            //The first dimensional is the channel
                            //The second dimension is the index of the rectangle in the specific channel                     
                            Rectangle[] facesDetected = face.DetectMultiScale(
                               ugray,
                               1.1,
                               10,
                               new Size(200, 200));

                            faces.AddRange(facesDetected);

                            foreach (Rectangle f in facesDetected)
                            {
                                //Get the region of interest on the faces
                                using (UMat faceRegion = new UMat(ugray, f))
                                {
                                    Rectangle[] eyesDetected = eye.DetectMultiScale(
                                       faceRegion,
                                       1.1,
                                       10,
                                       new Size(20, 20));

                                    foreach (Rectangle e in eyesDetected)
                                    {
                                        Rectangle eyeRect = e;
                                        eyeRect.Offset(f.X, f.Y);
                                        eyes.Add(eyeRect);
                                    }
                                }
                            }
                        }
                        watch.Stop();
                    }
                }
                detectionTime = watch.ElapsedMilliseconds;
            }
        }


        public static void Run(FaceParams MainFace, int SizeBufNum = 3, int BinaryValue = 143)
        {

            //Read the files as an 8-bit Bgr image  


            long detectionTime;
            List<Rectangle> faces = new List<Rectangle>();
            List<Rectangle> eyes = new List<Rectangle>();

            Detect(MainFace.CurrentFrame, @"Haar\haarcascade_frontalface_default.xml", @"Haar\haarcascade_eye.xml",
              faces, eyes,
              out detectionTime);

            if (faces.Count == 1)
            {

                if (Math.Abs(MainFace.PrevFaceSize.X - faces[0].X) < SizeBufNum && !MainFace.PrevFaceSize.IsEmpty)
                {
                    faces[0] = MainFace.PrevFaceSize;
                }
                CvInvoke.Rectangle(MainFace.CurrentFrame, faces[0], new Bgr(Color.Red).MCvScalar, 2);
                MainFace.FaceImg = new Mat((Mat)MainFace.CurrentFrame, faces[0]);
                MainFace.PrevFaceSize = faces[0];
            }

            if (eyes.Count == 2)

            {

                if (eyes[0].X < eyes[1].X) //0 - правый, 1- левый
                {
                    eyes.Reverse();
                }
                if (Math.Abs(MainFace.PrevRightEyeSize.X - eyes[0].X) < SizeBufNum && !MainFace.PrevRightEyeSize.IsEmpty)
                    eyes[0] = MainFace.PrevRightEyeSize;
                if (Math.Abs(MainFace.PrevLeftEyeSize.X - eyes[1].X) < SizeBufNum && !MainFace.PrevLeftEyeSize.IsEmpty)
                    eyes[1] = MainFace.PrevLeftEyeSize;

                MainFace.PrevRightEyeSize = eyes[0];
                MainFace.PrevLeftEyeSize = eyes[1];

                MainFace.RightEyeImg = new Mat((Mat)MainFace.CurrentFrame, eyes[0]);
                MainFace.LeftEyeImg = new Mat((Mat)MainFace.CurrentFrame, eyes[1]);

                CvInvoke.Rectangle(MainFace.CurrentFrame, eyes[0], new Bgr(Color.Blue).MCvScalar, 2);
                CvInvoke.Rectangle(MainFace.CurrentFrame, eyes[1], new Bgr(Color.Green).MCvScalar, 2);


                VectorOfVectorOfPoint RightPupilAreas =  PupilDetection.Detect(MainFace.RightEyeImg, BinaryValue, MainFace);
                VectorOfVectorOfPoint LeftPupilAreas = PupilDetection.Detect(MainFace.LeftEyeImg);
                
                for(int i = 0; i<RightPupilAreas.Size; i++)
                    CvInvoke.DrawContours(MainFace.RightEyeImg, RightPupilAreas, i, new MCvScalar(255, 0, 0));
                
                for (int i = 0; i < LeftPupilAreas.Size; i++)
                    CvInvoke.DrawContours(MainFace.LeftEyeImg, LeftPupilAreas, i, new MCvScalar(255, 0, 0));

                
            }
                   
        }

    }
}
