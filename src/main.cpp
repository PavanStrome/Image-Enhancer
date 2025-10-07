#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#ifndef NO_SUPERRES
#include <opencv2/dnn_superres.hpp>
#endif

#include <iostream>
#include <string>
#include <vector>

using std::cerr;
using std::cout;
using std::endl;

struct Args {
	std::string inputPath;
	std::string outputPath = "enhanced.png";
	std::string cascadePath = "haarcascade_frontalface_default.xml";
	std::string srModelPath = ""; // EDSR/LapSRN model file (.pb)
	double srScale = 2.0;
	double sharpenAmount = 1.0; // 0-3 typical
};

static void printUsage() {
	cout << "Usage: image_enhancer --input <path> [--output out.png] [--cascade haar.xml]" << endl;
#ifndef NO_SUPERRES
	cout << "       [--sr_model model.pb] [--sr_scale 2|3|4]" << endl;
#endif
	cout << "       [--sharpen 0..3]" << endl;
}

static bool parseArgs(int argc, char** argv, Args& args) {
	if (argc < 3) {
		printUsage();
		return false;
	}
	for (int i = 1; i < argc; ++i) {
		std::string key = argv[i];
		if (key == "--input" && i + 1 < argc) {
			args.inputPath = argv[++i];
		} else if (key == "--output" && i + 1 < argc) {
			args.outputPath = argv[++i];
		} else if (key == "--cascade" && i + 1 < argc) {
			args.cascadePath = argv[++i];
#ifndef NO_SUPERRES
		} else if (key == "--sr_model" && i + 1 < argc) {
			args.srModelPath = argv[++i];
		} else if (key == "--sr_scale" && i + 1 < argc) {
			args.srScale = std::stod(argv[++i]);
#endif
		} else if (key == "--sharpen" && i + 1 < argc) {
			args.sharpenAmount = std::stod(argv[++i]);
		} else if (key == "--help" || key == "-h") {
			printUsage();
			return false;
		}
	}
	if (args.inputPath.empty()) {
		cerr << "--input is required" << endl;
		return false;
	}
	return true;
}

static cv::Mat unsharpMask(const cv::Mat& srcBgr, double amount) {
	if (amount <= 0.0) return srcBgr.clone();
	cv::Mat blurred;
	int k = 0;
	// kernel size based on amount
	if (amount < 0.75) k = 3; else if (amount < 1.5) k = 5; else if (amount < 2.5) k = 7; else k = 9;
	cv::GaussianBlur(srcBgr, blurred, cv::Size(k, k), 0);
	cv::Mat sharp;
	cv::addWeighted(srcBgr, 1.0 + amount, blurred, -amount, 0, sharp);
	return sharp;
}

static cv::Mat enhanceLumaCLAHE(const cv::Mat& srcBgr) {
	cv::Mat ycrcb;
	cv::cvtColor(srcBgr, ycrcb, cv::COLOR_BGR2YCrCb);
	std::vector<cv::Mat> ch;
	cv::split(ycrcb, ch);
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
	clahe->apply(ch[0], ch[0]);
	cv::merge(ch, ycrcb);
	cv::Mat out;
	cv::cvtColor(ycrcb, out, cv::COLOR_YCrCb2BGR);
	return out;
}

static cv::Rect pickLargest(const std::vector<cv::Rect>& rects) {
	if (rects.empty()) return {};
	size_t best = 0;
	for (size_t i = 1; i < rects.size(); ++i) {
		if (rects[i].area() > rects[best].area()) best = i;
	}
	return rects[best];
}

static cv::Mat featherMask(const cv::Size& size, int radius) {
	cv::Mat mask(size, CV_32F, cv::Scalar(1.0f));
	int r = std::max(1, radius);
	cv::Mat kernel = cv::getGaussianKernel(r * 2 + 1, r, CV_32F);
	cv::Mat gauss;
	cv::sepFilter2D(mask, gauss, -1, kernel, kernel);
	cv::normalize(gauss, gauss, 0.0, 1.0, cv::NORM_MINMAX);
	return gauss;
}

static void pasteWithFeather(const cv::Mat& faceBgr, const cv::Rect& roi, cv::Mat& canvasBgr) {
	cv::Mat dstROI = canvasBgr(roi);
	cv::Mat resizedFace;
	cv::resize(faceBgr, resizedFace, roi.size(), 0, 0, cv::INTER_CUBIC);
	cv::Mat mask = featherMask(roi.size(), std::max(3, roi.width / 20));
	cv::Mat mask3;
	cv::Mat maskChannels[] = {mask, mask, mask};
	cv::merge(maskChannels, 3, mask3);
	cv::Mat dstF, srcF, mF;
	dstROI.convertTo(dstF, CV_32F, 1.0/255.0);
	resizedFace.convertTo(srcF, CV_32F, 1.0/255.0);
	mask3.convertTo(mF, CV_32F);
	cv::Mat blended = srcF.mul(mF) + dstF.mul(1.0 - mF);
	blended.convertTo(dstROI, CV_8U, 255.0);
}

int main(int argc, char** argv) {
	Args args;
	if (!parseArgs(argc, argv, args)) return 1;

	cv::Mat img = cv::imread(args.inputPath, cv::IMREAD_COLOR);
	if (img.empty()) {
		cerr << "Failed to read input image: " << args.inputPath << endl;
		return 2;
	}

	cv::CascadeClassifier faceCascade;
	if (!faceCascade.load(args.cascadePath)) {
		cerr << "Failed to load cascade: " << args.cascadePath << endl;
		return 3;
	}

	// Detect face
	std::vector<cv::Rect> faces;
	cv::Mat gray;
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(gray, gray);
	faceCascade.detectMultiScale(gray, faces, 1.2, 5, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(40, 40));
	cv::Rect faceRect = pickLargest(faces);
	if (faceRect.area() == 0) {
		cerr << "No face detected. Saving original to output." << endl;
		cv::imwrite(args.outputPath, img);
		return 0;
	}

	// Expand ROI slightly
	int padX = faceRect.width / 8;
	int padY = faceRect.height / 6;
	cv::Rect roi(std::max(0, faceRect.x - padX), std::max(0, faceRect.y - padY),
		std::min(faceRect.width + 2 * padX, img.cols - std::max(0, faceRect.x - padX)),
		std::min(faceRect.height + 2 * padY, img.rows - std::max(0, faceRect.y - padY)));

	cv::Mat face = img(roi).clone();

	// Optional super-resolution
#ifndef NO_SUPERRES
	if (!args.srModelPath.empty() && args.srScale >= 1.5) {
		try {
			cv::dnn_superres::DnnSuperResImpl sr;
			// Heuristic: infer algo from filename
			std::string lower = args.srModelPath;
			for (auto& c : lower) c = (char)std::tolower(c);
			std::string algo = lower.find("edsr") != std::string::npos ? "edsr" :
				(lower.find("lapsrn") != std::string::npos ? "lapsrn" : "edsr");
			sr.readModel(args.srModelPath);
			sr.setModel(algo, (int)std::round(args.srScale));
			cv::Mat up;
			sr.upsample(face, up);
			face = up;
		} catch (const std::exception& e) {
			cerr << "Super-resolution failed: " << e.what() << ". Using bicubic." << endl;
			cv::Mat up;
			cv::resize(face, up, cv::Size(), args.srScale, args.srScale, cv::INTER_CUBIC);
			face = up;
		}
	} else
#endif
	{
		if (args.srScale > 1.01) {
			cv::Mat up;
			cv::resize(face, up, cv::Size(), args.srScale, args.srScale, cv::INTER_CUBIC);
			face = up;
		}
	}

	// Sharpen and contrast enhance
	face = unsharpMask(face, args.sharpenAmount);
	face = enhanceLumaCLAHE(face);

	// Mild denoise if too grainy after CLAHE
	cv::Mat denoised;
	cv::fastNlMeansDenoisingColored(face, denoised, 3, 3, 7, 21);
	face = denoised;

	// Resize back down to ROI size for blending
	cv::Mat faceBack;
	cv::resize(face, faceBack, roi.size(), 0, 0, cv::INTER_CUBIC);

	cv::Mat result = img.clone();
	pasteWithFeather(faceBack, roi, result);

	if (!cv::imwrite(args.outputPath, result)) {
		cerr << "Failed to write output: " << args.outputPath << endl;
		return 4;
	}
	cout << "Saved: " << args.outputPath << endl;
	return 0;
}
