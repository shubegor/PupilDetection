using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PupilApp.Face
{
    public class PupilDetection
    {
        public static VectorOfVectorOfPoint Detect(IImage eye, int BinaryValue = 140, FaceParams MainFace = null)
        {
            
            Image<Gray, Byte> _eye = new Image<Gray, Byte>(eye.Bitmap);
            _eye = _eye.ThresholdBinary(new Gray(BinaryValue), new Gray(255));
            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            Mat Heir = new Mat();
            CvInvoke.FindContours(_eye, contours, Heir, RetrType.External, ChainApproxMethod.ChainApproxSimple);
            VectorOfVectorOfPoint pupilAreas = new VectorOfVectorOfPoint();
            if (MainFace != null) MainFace.RightEyeImg = _eye;

            for (int i = 0; i < contours.Size; i++)
            {
                double area = CvInvoke.ContourArea(contours[i]);

                if (area < 50 && area > 5)
                {
                    pupilAreas.Push(contours[i]);
                }

            }
            return pupilAreas;
        }
    }
}
