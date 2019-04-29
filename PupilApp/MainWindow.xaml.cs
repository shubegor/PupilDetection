using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Media;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.Util;
using Emgu.CV.UI;
using Emgu.CV.Cuda;
using Emgu.CV.Util;
using Emgu.CV.WPF;
using System.Globalization;
using System.ComponentModel;

namespace PupilApp
{
    /// <summary>
    /// Логика взаимодействия для MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window, INotifyPropertyChanged
    {
        private VideoCapture _capture = null;
        private bool _captureInProgress;
        private Mat _frame;
        private Mat _grayFrame;
        private Mat _smallGrayFrame;
        private Mat _smoothedGrayFrame;
        FaceParams MainFace = new FaceParams();
        public int Threshold
        {
            get { return _threshold; }
            set { _threshold = value; this.OnPropertyChanged("Threshold"); }
        }
        private int _threshold;
        public MainWindow()
        {
            InitializeComponent();
            Threshold = 127;
            CvInvoke.UseOpenCL = false;
            try
            {
                _capture = new VideoCapture();
                _capture.ImageGrabbed += ProcessFrame;

            }
            catch (NullReferenceException excpt)
            {
                MessageBox.Show(excpt.Message);
            }
            _frame = new Mat();
            _grayFrame = new Mat();
            _smallGrayFrame = new Mat();
            _smoothedGrayFrame = new Mat();
        }

        private void ProcessFrame(object sender, EventArgs arg)
        {
            if (_capture != null && _capture.Ptr != IntPtr.Zero)
            {
                _capture.Retrieve(_frame, 0);

                CvInvoke.CvtColor(_frame, _grayFrame, ColorConversion.Bgr2Gray);

                CvInvoke.PyrDown(_grayFrame, _smallGrayFrame);

                CvInvoke.PyrUp(_smallGrayFrame, _smoothedGrayFrame);


                MainFace.CurrentFrame = _frame;
                Face.DetectFace.Run(MainFace, 3, Threshold);


                BitmapSource bi = BitmapSourceConvert.ToBitmapSource(MainFace.CurrentFrame);
                bi.Freeze();
                Dispatcher.BeginInvoke(new ThreadStart(delegate { MainVideo.Source = bi; }));

                if(MainFace.FaceImg!=null)
                {
                    BitmapSource face = BitmapSourceConvert.ToBitmapSource(MainFace.FaceImg);
                    face.Freeze();
                    Dispatcher.BeginInvoke(new ThreadStart(delegate { FaceVideo.Source = face; }));
                }

                if (MainFace.LeftEyeImg != null)
                {
                    BitmapSource LEye = BitmapSourceConvert.ToBitmapSource(MainFace.LeftEyeImg);
                    LEye.Freeze();
                    Dispatcher.BeginInvoke(new ThreadStart(delegate { LEyeVideo.Source = LEye; }));
                }
                if (MainFace.RightEyeImg != null)
                {
                    BitmapSource REye = BitmapSourceConvert.ToBitmapSource(MainFace.RightEyeImg);
                    REye.Freeze();
                    Dispatcher.BeginInvoke(new ThreadStart(delegate { REyeVideo.Source = REye; }));
                }
                //FaceImageBox.Image = res.FaceImg;
                //LeftEyeImageBox.Image = res.LeftEyeImg;
                //RightEyeBox.Image = res.RightEyeImg;
            }
        }
        

        private void Camera_Click(object sender, RoutedEventArgs e)
        {
            if (_capture != null)
            {
                if (_captureInProgress)
                {  //stop the capture
                    //captureButton.Text = "Start Capture";
                    _capture.Pause();
                }
                else
                {
                    //start the capture
                    //captureButton.Text = "Stop";
                    _capture.Start();
                }

                _captureInProgress = !_captureInProgress;
            }
        }

        #region INotifyPropertyChanged members

        public event PropertyChangedEventHandler PropertyChanged;

        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChangedEventHandler handler = this.PropertyChanged;
            if (handler != null)
            {
                var e = new PropertyChangedEventArgs(propertyName);
                handler(this, e);
            }
        }

        #endregion
    }
}
