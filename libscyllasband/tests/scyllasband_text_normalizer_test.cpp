#include "scyllasband_text_normalizer.h"

#include <iostream>
#include <string>

namespace {

int expect_equal(
    const std::string& language,
    const std::string& input,
    const std::string& expected
) {
    const std::string actual = scyllasband_detail::normalize_spoken_text(input, language);
    if (actual == expected) {
        return 0;
    }
    std::cerr << "normalizer mismatch for " << language << "\n"
              << "expected: " << expected << "\n"
              << "actual:   " << actual << "\n";
    return 1;
}

}  // namespace

int main() {
    int failures = 0;
    failures += expect_equal(
        "en_us",
        "March 15, 2024 cost $47,500; at 3:05 PM it was 14.5% and 3/4 done.",
        "March fifteenth, two thousand twenty four cost forty seven thousand five hundred dollars; "
        "at three oh five PM it was fourteen point five percent and three fourths done."
    );
    failures += expect_equal(
        "en_gb",
        "March 15, 2024 cost £47.50; at 3:05 PM it was 14.5% and 3/4 done.",
        "March fifteenth, two thousand twenty four cost forty seven pounds and fifty pence; "
        "at three oh five PM it was fourteen point five percent and three fourths done."
    );
    failures += expect_equal(
        "es",
        "15/03/2024 costó €47,50; a las 3:05 estaba al 14,5% y 3/4 hecho.",
        "quince de marzo de dos mil veinticuatro costó cuarenta y siete euros y cincuenta céntimos; "
        "a las tres cinco estaba al catorce coma cinco por ciento y tres sobre cuatro hecho."
    );
    failures += expect_equal(
        "it",
        "15/03/2024 costava €47,50; alle 3:05 era al 14,5% e 3/4 fatto.",
        "quindici marzo due mila venti quattro costava quaranta sette euro e cinquanta centesimi; "
        "alle tre cinque era al quattordici virgola cinque per cento e tre su quattro fatto."
    );
    failures += expect_equal(
        "fr_fr",
        "Le prix est €21,50 et 5%.",
        "Le prix est vingt et un euros et cinquante centimes et cinq pour cent."
    );
    failures += expect_equal(
        "fr",
        "Le 3o essai et le 80o.",
        "Le troisième essai et le quatre vingtième."
    );
    failures += expect_equal(
        "de_de",
        "Es kostet €21,50 und 5%.",
        "Es kostet einundzwanzig Euro und fünfzig Cent und fünf Prozent."
    );
    failures += expect_equal(
        "vi_vn",
        "Giá là €21,50 và 5%.",
        "Giá là hai mươi mốt euro và năm mươi xu và năm phần trăm."
    );
    failures += expect_equal(
        "en_us",
        "D.J. played the A-game. It grossed $47500, cost $99.99, and moved at 0.03 miles per hour.",
        "dee jay played the ay game. "
        "It grossed forty seven thousand five hundred dollars, cost ninety nine dollars and ninety nine cents, "
        "and moved at zero point zero three miles per hour."
    );
    return failures == 0 ? 0 : 1;
}
