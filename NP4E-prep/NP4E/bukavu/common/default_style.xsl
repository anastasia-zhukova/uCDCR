<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0"
xmlns:mmax="org.eml.MMAX2.discourse.MMAX2DiscourseLoader"
xmlns:section="www.eml.org/NameSpaces/section"
xmlns:case="www.eml.org/NameSpaces/case"
xmlns:chunk="www.eml.org/NameSpaces/chunk"
xmlns:response="www.eml.org/NameSpaces/response"
xmlns:enamex="www.eml.org/NameSpaces/enamex"
xmlns:markable="www.eml.org/NameSpaces/markable"
xmlns:morph="www.eml.org/NameSpaces/morph"
xmlns:sentenceclean="www.eml.org/NameSpaces/sentenceclean"
xmlns:sentence="www.eml.org/NameSpaces/sentence"
xmlns:semrole="www.eml.org/NameSpaces/semrole"
xmlns:parse="www.eml.org/NameSpaces/parse"
xmlns:coref="www.eml.org/NameSpaces/coref"
xmlns:pos="www.eml.org/NameSpaces/pos"
xmlns:doc="www.eml.org/NameSpaces/doc"
xmlns:topic="www.eml.org/NameSpaces/topic"
xmlns:paragraph="www.eml.org/NameSpaces/paragraph">
<xsl:output method="text" indent="no" omit-xml-declaration="yes"/>
<xsl:strip-space elements="*"/>

<xsl:template match="words">
  <xsl:apply-templates/>
</xsl:template>

<xsl:template match="word">
 <xsl:value-of select="mmax:registerDiscourseElement(@id)"/>
  <xsl:apply-templates select="mmax:getStartedMarkables(@id)" mode="opening"/>
  <xsl:value-of select="mmax:setDiscourseElementStart()"/>
   <xsl:apply-templates/>
  <xsl:value-of select="mmax:setDiscourseElementEnd()"/>
  <xsl:apply-templates select="mmax:getEndedMarkables(@id)" mode="closing"/>
<xsl:text> </xsl:text>
</xsl:template>

<!--  TOPIC: 1 EMPTY LINE + MARKER SEPARATED -->

<xsl:template match="topic:markable" mode="closing">
<xsl:text>

------------------------------ TOPIC ENDS -----------------------------
</xsl:text>
</xsl:template>

<!--  DOC: 1 EMPTY LINE + MARKER SEPARATED -->

<xsl:template match="doc:markable" mode="closing">
<xsl:text>

---------------------------- DOCUMENT ENDS ----------------------------
</xsl:text>
</xsl:template>

<!--  PARAGRAPH: 1 EMPTY LINE SEPARATED -->

<xsl:template match="paragraph:markable" mode="opening">
<xsl:text>
</xsl:text>
</xsl:template>

<!--  SENTENCE CLEAN: 1 SENTENCE PER LINE -->
<xsl:template match="sentenceclean:markable" mode="opening">
<xsl:text>
</xsl:text>
</xsl:template>

</xsl:stylesheet>
