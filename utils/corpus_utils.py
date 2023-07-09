def extract_early_life(page):
    sections = page.sections

    # Check if the page has sections
    if sections:
        extracted_text = ""
        # Iterate over the sections
        for section in sections:
            # Look for the desired sections
            if any(keyword in section.title.lower() for keyword in ['early', 'life', 'family', 'childhood', 'infancy', 'education', 'biography', 'background', 'career', 'profession', 'professional']):
                # Extract the text from the section
                section_text = section.text.strip()
                if section_text:
                    extracted_text += section_text + "\n"

                # Extract the subsections
                for sub_section in section.sections:
                    if any(keyword in sub_section.title.lower() for keyword in ['early', 'life', 'family', 'childhood', 'infancy', 'education', 'biography', 'background', 'career', 'profession', 'professional']):
                        sub_section_text = sub_section.text.strip()
                        if sub_section_text:
                            extracted_text += sub_section_text + "\n"

                        # Extract the sub_subsections
                        for sub_sub_section in sub_section.sections:
                            if any(keyword in sub_sub_section.title.lower() for keyword in ['early', 'life', 'family', 'childhood', 'infancy', 'education', 'biography', 'background', 'career', 'profession', 'professional']):
                                sub_sub_section_text = sub_sub_section.text.strip()
                                if sub_sub_section_text:
                                    extracted_text += sub_sub_section_text + "\n"

        # If the desired sections and subsections are found, return the extracted text
        if extracted_text:
            return extracted_text

    # If the desired sections and subsections are not found, return None
    return None