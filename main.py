import sys
import re
import jieba
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FileHandler:
    """文件处理类，负责文件的读取和写入操作"""

    @staticmethod
    def read_file(file_path):
        """读取文件内容"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'ansi']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue

        # 如果所有编码都尝试失败，抛出异常
        raise UnicodeDecodeError(f"无法解析文件 {file_path}，请检查文件编码")

    @staticmethod
    def write_file(file_path, content):
        """写入内容到文件"""
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)


class TextProcessor:
    """文本处理类，负责文本预处理和分词"""

    def __init__(self):
        """初始化文本处理器，加载停用词"""
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self):
        """加载停用词列表"""
        # 内置停用词列表
        stopwords_list = [
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
            '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有',
            '看', '好', '自己', '这', '个', '以', '他', '还', '而', '后', '之', '来',
            '及', '于', '其', '与', '所', '对', '也', '但', '并', '或', '且', '着', '了',
            '过', '呢', '吗', '吧', '啊', '呀', '啦', '之', '乎', '者', '也', '矣', '焉', '哉'
        ]
        return set(stopwords_list)

    def preprocess(self, text):
        """预处理文本：去除特殊字符、HTML标签等"""
        # 去除HTML标签
        text = re.sub(r'<.*?>', '', text)
        # 去除特殊字符和标点
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def segment(self, text):
        """中文分词并去除停用词"""
        # 使用jieba进行分词
        words = jieba.cut(text)
        # 去除停用词
        filtered_words = [word for word in words if word not in self.stopwords and word.strip()]
        # 用空格连接词语，形成适合TF-IDF处理的格式
        return ' '.join(filtered_words)


class SimilarityCalculator:
    """相似度计算类，负责计算两篇文本的相似度"""

    @staticmethod
    def calculate(text1, text2):
        """计算两篇文本的相似度"""
        # 处理空文本情况
        if not text1.strip() and not text2.strip():
            return 1.0  # 两篇都是空文本，相似度为1
        if not text1.strip() or not text2.strip():
            return 0.0  # 一篇为空，相似度为0

        # 创建TF-IDF向量器
        vectorizer = TfidfVectorizer()
        # 拟合并转换文本为TF-IDF矩阵
        tfidf_matrix = vectorizer.fit_transform([text1, text2])

        # 计算余弦相似度
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

        return cosine_sim[0][0]


def main():
    try:
        # 解析命令行参数
        parser = argparse.ArgumentParser(description='计算文本相似度（检测抄袭率）')
        parser.add_argument('original', help='原文文件路径')
        parser.add_argument('copy', help='抄袭版文件路径')
        parser.add_argument('result', help='结果输出文件路径')
        args = parser.parse_args()

        # ---------- 步骤1：打印程序启动 ----------
        print("=== 程序开始执行 ===")

        # ---------- 步骤2：显示文件路径 ----------
        orig_path = args.original
        copy_path = args.copy
        result_path = args.result

        print(f"原文路径: {orig_path}")
        print(f"抄袭版路径: {copy_path}")
        print(f"结果路径: {result_path}")

        # ---------- 步骤3：读取文件 ----------
        file_handler = FileHandler()
        print(" 开始读取原文...")
        orig_text = file_handler.read_file(orig_path)
        print(f" 原文读取成功，长度: {len(orig_text)}")

        print(" 开始读取抄袭版论文...")
        copy_text = file_handler.read_file(copy_path)
        print(f" 抄袭版读取成功，长度: {len(copy_text)}")

        # ---------- 步骤4：文本预处理与分词 ----------
        text_processor = TextProcessor()
        print(" 开始预处理原文...")
        orig_processed = text_processor.preprocess(orig_text)

        print(" 开始预处理抄袭版论文...")
        copy_processed = text_processor.preprocess(copy_text)

        print(" 开始对原文分词...")
        orig_segmented = text_processor.segment(orig_processed)
        print(f" 原文分词结果（前10个词）: {orig_segmented.split()[:10]}")

        print(" 开始对抄袭版论文分词...")
        copy_segmented = text_processor.segment(copy_processed)

        # ---------- 步骤5：计算相似度 ----------
        calculator = SimilarityCalculator()
        print(" 开始计算相似度...")
        similarity = calculator.calculate(orig_segmented, copy_segmented)
        print(f" 相似度计算完成，原始值: {similarity}")

        # 保留两位小数
        similarity_rounded = round(similarity, 2)
        print(f"️ 保留两位小数后: {similarity_rounded}")

        # ---------- 步骤6：写入结果文件 ----------
        print(" 开始写入结果文件...")
        file_handler.write_file(result_path, str(similarity_rounded))
        print("️ 结果写入成功！")

    except Exception as e:
        # 打印详细异常
        import traceback
        print(f" 发生错误: {str(e)}", file=sys.stderr)
        print(" 详细错误堆栈:\n", traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()