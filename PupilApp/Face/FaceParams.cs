using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using System.Drawing;

namespace PupilApp
    {
    public class FaceParams
    {
        public IImage CurrentFrame { get; set; }
        public IImage FaceImg { get; set; }
        public Rectangle PrevFaceSize { get; set; }



        public IImage LeftEyeImg { get; set; }
        public double LeftPupilArea { get; set; }
        public Rectangle PrevLeftEyeSize { get; set; }



        public IImage RightEyeImg { get; set; }
        public double RightPupilArea { get; set; }
        public Rectangle PrevRightEyeSize { get; set; }
    }
}
