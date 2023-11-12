from Bio import Align
from Bio.Seq import Seq


def merge_multiple_sequences(*sequences):
    aligner = Align.PairwiseAligner()

    # Biopython Seqオブジェクトに変換
    seq_objects = [Seq(seq) for seq in sequences]

    # 一つ目のシーケンスと他のシーケンスを順番にアラインメントしてマージ
    merged_seq = seq_objects[0]
    for seq in seq_objects[1:]:
        alignments = aligner.align(merged_seq, seq)
        if alignments:
            aligned_seq1, aligned_seq2 = alignments[0].aligned
            overlap_start = aligned_seq1[0]
            overlap_end = aligned_seq1[-1] + 1  # endはスライスに含まれないので+1
            merged_seq = merged_seq[:overlap_start] + seq[overlap_end:]
        else:
            # アラインメントが見つからなかった場合、単純に連結
            merged_seq += seq

    return str(merged_seq)


# テスト
text1 = "GGGGHHHko"
text2 = "kko.code"
text3 = "ko.code"
output_union = [
    "plejjjGq_box_0.15mm ",
    "lejjjGq_box_0.15mm_ ",
    "ejjjGq_box_0.15mm_P ",
    "jjjGq_box_0.15mm_Pl ",
    "jGq_box_0.15mm_PLA_ ",
    "Gq_box_0Ll5mm_PLA_M ",
    "x_0.l5mm_PLA_MK3_17 ",
]
result = output_union[0]
result = result.strip()
for i, text in enumerate(output_union[1:]):
    print(result[i + 1 :])
    py = text.strip()
    print("py")
    print(py)
    print("jokyo")
    print(py.replace(result[i + 1 :], ""))
    if py.replace(result[i + 1 :], "") != py:
        result += py.replace(result[i + 1 :], "")
    print(result)
print(result)
