"""Select housing referrals from a post without changing model predictions."""
import re

LEGAL = [
    ("Legal Services Corporation · find legal aid", "https://www.lsc.gov/about-lsc/what-legal-aid/i-need-legal-help", "Search by city or address for an LSC-funded civil legal aid organization. Ask whether it handles eviction defense, landlord disputes or housing conditions, and whether you qualify for free assistance. If you have a notice or hearing date, mention it when requesting intake."),
    ("LawHelp · legal help by state", "https://www.lawhelp.org/", "Find your state’s legal-help directory and look for tenant rights, eviction and housing resources. Local providers can explain which legal services are available; contact them about eligibility and any approaching court deadlines."),
    ("USAGov · emergency rent assistance", "https://www.usa.gov/emergency-pay-rent", "If unpaid rent is part of the problem, follow this guide to rental assistance contacts. Ask about current funding and application requirements while separately seeking legal help for any notice or court case."),
]
AFFORDABLE = [
    ("USAGov · subsidized low-rent apartments", "https://www.usa.gov/subsidized-rental-housing", "Find privately owned housing where subsidies reduce rent for eligible households. Follow the guide to HUD’s housing map, search your city, and contact individual properties about low-rent units, income limits, total move-in costs and open applications. Some properties have waiting lists."),
    ("USAGov · public housing", "https://www.usa.gov/public-housing", "Explore affordable rental homes managed by local public housing agencies. Use the agency finder to ask about eligibility, available properties and application steps. Compare the rent and required deposits with your budget; availability and waiting lists vary."),
    ("USAGov · Section 8 housing vouchers", "https://www.usa.gov/housing-voucher-section-8", "Vouchers can help eligible households pay rent in qualifying private housing. Contact your local public housing agency to check application openings and requirements. Waiting lists are common, so treat this as a longer-term option alongside direct applications to affordable properties."),
]
ARREARS = [
    ("USAGov · emergency rent assistance", "https://www.usa.gov/emergency-pay-rent", "Find contacts for emergency rental assistance if you are behind on rent. Ask whether applications are open, which expenses can be covered and what documents are needed. Funding and eligibility vary by program."),
    ("CFPB · help paying rent and bills", "https://www.consumerfinance.gov/housing/housing-insecurity/help-for-renters/get-help-paying-rent-and-bills/", "Review assistance options and links to housing counseling. Use the guidance to identify a provider for your situation and ask about application steps; if you have also received an eviction notice, seek legal aid promptly."),
    ("United Way 211 · local assistance finder", "https://211.org/about-us/your-local-211", "Find the 211 service covering your community and ask specifically about rent arrears or eviction prevention. Explain any deadline and ask which programs are currently taking requests and what paperwork they require."),
]
SHELTER = [
    ("HUD · Find Shelter", "https://www.hud.gov/FindShelter", "Search for shelter and services near your current location. Contact a provider about available space, intake times and any household or accessibility requirements before traveling; a listing does not guarantee an open bed."),
    ("USAGov · emergency housing", "https://www.usa.gov/emergency-housing-assistance", "Find next steps for people facing or experiencing homelessness. Follow the emergency housing links and contact the relevant service to explain when you need somewhere to stay and who needs accommodation."),
    ("United Way 211 · shelter referrals", "https://211.org/about-us/your-local-211", "Find your local 211 service and ask about shelter intake and homelessness services. Describe whether you need a place tonight, whether children or pets are with you, and any access needs so staff can help identify appropriate referrals."),
]


def housing_sections(text):
    """Return all relevant subtopics; explicit negative phrases suppress a match."""
    value = (text or "").lower().replace("’", "'")
    value = re.sub(r"\b(?:not facing|no risk of|not at risk of) (?:an? )?eviction\b", "", value)
    sections = []
    legal = re.search(r"\beviction\b|\bevict(?:ed|ing)?\b|tenant rights|housing court|landlord.{0,35}(?:threat|lock|refus|discriminat|deposit|mold|mould|repair)|changed (?:the |my )?locks|security deposit.{0,30}(?:return|withheld)|housing.{0,20}(?:lawyer|legal)|(?:lawyer|legal).{0,20}(?:housing|landlord|rent)", value)
    affordable = re.search(r"affordable (?:housing|apartment|rent|place|home)|(?:cheap|cheaper|low[ -](?:cost|income|rent)|subsidized).{0,25}(?:housing|apartment|rent|place|home)|section ?8|housing voucher|public housing|(?:find|looking for|searching for).{0,45}(?:apartment|rental|place to live)|rent.{0,20}(?:too high|too expensive|unaffordable)", value)
    arrears = re.search(r"behind on (?:my |the )?rent|rent arrears|(?:can't|cannot|unable to).{0,15}(?:pay|afford).{0,15}rent|(?:pay|paying).{0,15}(?:my |the )?rent|rental assistance|rent assistance", value)
    shelter = re.search(r"\bhomeless\b|\bshelter\b|sleeping in (?:my|a|the) car|nowhere to (?:sleep|stay|go)|no place to (?:sleep|stay)|living (?:on the street|in (?:my|a) car)", value)
    if legal:
        sections.append(("HOUSING · LEGAL AID & TENANT RIGHTS", LEGAL))
    if shelter:
        sections.append(("HOUSING · SHELTER & IMMEDIATE ACCOMMODATION", SHELTER))
    if arrears:
        sections.append(("HOUSING · HELP WITH RENT PAYMENTS", ARREARS))
    if affordable:
        sections.append(("HOUSING · AFFORDABLE RENTAL OPTIONS", AFFORDABLE))
    return sections
