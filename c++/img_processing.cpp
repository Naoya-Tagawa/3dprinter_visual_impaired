#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <math.h>
#include <cmath>
/*Projection profileから横方向の座標を得る*/
std::vector<double> Detect_WidthPosition(double W_THRESH, double width, double array_V[]){
    std::vector<double> char_List{};
    bool flg = false;
    double posi1 = 0;
    double posi2 = 0;
    
    for (int i =0;i<width;i++){
        double val = array_V[i];
        if ((flg==false) && (val < W_THRESH)){
            bool flg = true;
            posi1 = i;
        }
        if ((flg == true) && (val >= W_THRESH)){
            flg = false;
            posi2 = i;
            char_List.push_back(posi1);
            char_List.push_back(posi2);
        }
    }   
    return char_List;
}
/*Projection profileから縦方向の座標を得る*/
std::vector<double> Detect_HeightPosition(double H_THRESH, double height, double array_H[]){
    /*char_List = np.array([])*/
    /*char_Listは戻り値のための配列*/
    std::vector<double> char_List{};
 
    bool flg = false;
    double posi1 = 0;
    double posi2 = 0;
    for (int i = 0; i < height;i++){
        double val = array_H[i];
        if ( (flg==false) && (val < H_THRESH)){
            flg = true;
            posi1 = i;
        }
 
        if ((flg == true) && (val >= H_THRESH)){
            flg = false;
            posi2 = i;
            char_List.push_back(posi1);
            char_List.push_back(posi2);
        }
    }
    return char_List;
}
/*#縦方向のProjection profileを得る*/
std::vector<double> Projection_H(int height, int width,std::vector<std::vector<double>> img){
    std::vector<double> array_H(height,0);
    /*heightは要素数,0は初期化する値*/
    for (int i = 0; i< height;i++){
        int total_count = 0;
        for (int j = 0; j < width;j++){
            double temp_pixval = img[i][j];
            if (temp_pixval == 0){
                total_count = total_count + 1;

            }
        }
        array_H[i] = total_count;
    }
    return array_H;
}
/*横方向のProjection profileを得る*/
std::vector<double> Projection_V(int height, int width, std::vector<std::vector<double>> img){
    std::vector<double> array_V(width,0);
    for (int i = 0; i< width; i++){
        int total_count = 0;
        for (int j = 0; j < height; j++){
            double temp_pixVal = img[j][i];
            if (temp_pixVal == 0){
                total_count = total_count + 1;
            }
        }
        array_V[i] = total_count;
    }
    return array_V;
}
std::vector<double> projective_transformation(std::vector<std::vector<double>> img1,double p1[],double p2[],double p3[],double p4[]){
    /*#座標*/
    /*p1 左上*/
    /*p2 左下*/
    /*p3 右上*/
    /*p4 右下*/
    /*幅*/
    double w = std::abs(p3[0]-p1[0]);
    w = floor(w);
    double h = std::abs(p2[1]-p1[1]);
    h = floor(h);
    double data[] = {p1[0],p1[1],p3[0],p3[1],p2[0],p2[1],p4[0],p4[1]};
    double data1[] = {0,0,w,0,0,h,w,h};
    double data2[2] = {w,h};
    /*pts1はカードの4辺、pts2は変換後の座標*/
    cv::Mat pts1 (2,4,CV_32F,data);
    cv::Mat pts2 (2,4,CV_32F,data1);
    /*std::vector<std::vector<double>> pts1 {{p1[0],p1[1]},{p3[0],p3[1]},{p2[0],p2[1]},{p4[0],p4[1]}};*/
    /*std::vector<std::vector<double>> pts2 {{0,0},{w,0},{0,h},{w,h}};*/
    /*射影変換を実施*/

    std::vector<std::vector<double>> M;
    std::vector<std::vector<double>> dst;
    cv::Mat M;
    cv::Mat dst;
    cv::Mat st (1,1,CV_32F,data2);
    M = cv::getPerspectiveTransform(pts1, pts2);
    /*dst = cv::warpPerspective(img1, M, st,img1.size(),cv::INTER_LINEAR);*/
    /*return dst;*/
}

void imread(std::vector<double>img){
    cv::imshow("title",img);
    cv::waitKey(0);
    


}

int main( int argc, char** argv )
{

    printf("Hello World.\n");
    std::vector<double> img = cv::imread("./camera1/camera1.jpg");
    return 0;
}